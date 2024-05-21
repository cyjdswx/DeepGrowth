"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os,sys
from re import A
import argparse
from datetime import datetime
import time
from einops import rearrange, repeat
#from models.longmri_model import LongMri4DGenerator 
#from models.longmripnt_model import LongMripntModel
from models.networks.convlstm import Conv3DLSTM, STConv3DLSTMSmall, STConv3DLSTM
import torch
import nibabel as nib
import numpy as np
from util.util import pad_nd_image, compute_steps_for_sliding_window, get_gaussian,load_network
from skimage.measure import label

def getlargestcc(segmentation):
    labels = label(segmentation)
    #assert(labels.max() != 0)
    if labels.max() == 0:
        print('no segment')
        return np.ones_like(labels)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC

def get_relativedates(scans, maximal_period=96):
    study_dates = [datetime.strptime(i,'%Y%m%d') for i in scans]
    normalized_dates = []
    baseline_date = study_dates[0]
    for study_date in study_dates:
        day_diff = study_date.day - baseline_date.day
        month_diff = study_date.month - baseline_date.month
        year_diff = study_date.year - baseline_date.year
        #self.study_dates.append(year_diff * 12 + month_diff)
        time_diff = year_diff * 12 + month_diff + float(day_diff) / 30
        if time_diff > maximal_period - 1:
            time_diff = maximal_period - 1
        normalized_dates.append(time_diff)
    normalized_dates = np.array(normalized_dates)
    normalized_dates = normalized_dates / maximal_period 
    return normalized_dates

def create_nonzero_mask(data):
    from scipy.ndimage import binary_fill_holes
    assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        this_mask = data[c] != -1
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask

def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # experiment specifics
    parser.add_argument('--name', type=str, default='label2coco', help='name of the experiment')
    parser.add_argument('--config_file', type=str,default='./configs/brats.json')
    parser.add_argument('--nThreads', default=12, type=int, help='# threads for loading data')
    parser.add_argument('--load_from_opt_file', action='store_true', help='load the options from checkpoints and use that as default')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    
    parser.add_argument('--model', type=str, default='pix2pix', help='which model to use')
    parser.add_argument('--norm_G', type=str, default='instanceaffine', help='instance normalization or batch normalization')
    parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')
    parser.add_argument('--norm_E', type=str, default='spectralinstance', help='instance normalization or batch normalization')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

    # input/output sizes
    parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
    parser.add_argument('--input_nc', type=int, default=2, help='# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.')
    parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')
    parser.add_argument('--patch_height', type=int, default=64, help='# of height of image patch')
    parser.add_argument('--patch_width', type=int, default=64, help='# of width of image patch')
    parser.add_argument('--patch_thickness', type=int, default=16, help='# of slices of image patch')

    # Hyperparameters
    parser.add_argument('--maximal_period', type=float, default=96, help='# of months')
    parser.add_argument('--lr_width', type=int, default=64, help='low res stream strided conv number of channles')
    parser.add_argument('--lr_max_width', type=int, default=1024, help='low res stream conv number of channles')
    parser.add_argument('--lr_depth', type=int, default=7, help='low res stream number of conv layers')
    parser.add_argument('--hr_width', type=int, default=64, help='high res stream number of MLP channles')
    parser.add_argument('--hr_depth', type=int, default=5, help='high res stream number of MLP layers')
    parser.add_argument('--latent_dim', type=int, default=64, help='high res stream number of MLP layers')
    parser.add_argument('--reflection_pad', action='store_true', help='if specified, use reflection padding at lr stream')
    parser.add_argument('--replicate_pad', action='store_true', help='if specified, use replicate padding at lr stream')
    parser.add_argument('--netG', type=str, default='ASAPNets', help='selects model to use for netG')
    parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')
    parser.add_argument('--hr_coor', choices=('cosine', 'None','siren'), default='cosine')

    parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')

    parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--fold', type=int, help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        
    parser.set_defaults(serial_batches=True)
    parser.set_defaults(phase='test')
    
    opt = parser.parse_args()
    opt.isTrain = False   # train or test
    fold = opt.fold
    save_dir = opt.results_dir
    data_dir = '/exports/lkeb-hpc/ychen/01_data/08_VS_followup/longitudinal_t1ce_whole_cropped64'
    device = torch.device('cpu' if opt.gpu_ids == -1 else 'cuda')
    patch_size = (64, 64, 64)
    #model = LongMri4DGenerator(opt, num_input=2, use_sine=True)
    #model = MultiMripntModel(opt, device) 
    Generator = Conv3DLSTM(input_dim=2,hidden_dim=[64,128,64],kernel_size=[(3,3,3),(3,3,3),(3,3,3)],num_layers=3,\
                batch_first=True,bias=True,return_all_layers=False)
    #Generator = STConv3DLSTM(input_dim=2,hidden_dim=[64,64,64],kernel_size=[(3,3,3),(3,3,3),(3,3,3)],num_layers=3,\
    #            batch_first=True,bias=True,return_all_layers=False)
    Generator.cuda()
    Generator = load_network(Generator, 'Generator', opt.which_epoch, opt)
    Generator.eval()

    listfile = 'patientlist_fold' + str(fold) + '_val.txt'
    with open(os.path.join(data_dir,listfile)) as f:
        patientlist = f.read().splitlines()
    device = torch.device('cuda')
    img_affine = None
    for patient in patientlist:
        print(patient)
        image_list = []
        study_dates = []
        patient_dir = os.path.join(data_dir, patient)
        patient_save_dir = os.path.join(save_dir, patient)
        patient_id = patient.split('_')[1]
        scans = os.listdir(patient_dir)
        scans = [s.split('_')[1] for s in scans if s.endswith('_seg.nii.gz')]
        scans.sort()
        print(scans)
        study_dates = get_relativedates(scans)
        scans_fur = scans[2:]
        scans_study = scans[:2]
        img_list = []
        seg_list = []
        image_handle = nib.load(os.path.join(patient_dir, patient_id + '_' + scans[0] + '.nii.gz'))
        img_affine = image_handle.affine
        for scan in scans:
            img_handle = nib.load(os.path.join(patient_dir, patient_id + '_' + scan + '.nii.gz'))
            img = np.asarray(img_handle.dataobj)
            img = np.transpose(img, (2,1,0))
            img = img * 2 - 1
            img_list.append(np.expand_dims(img,axis=0))
            
            seg_handle = nib.load(os.path.join(patient_dir, patient_id + '_' + scan + '_seg.nii.gz'))
            seg = np.asarray(seg_handle.dataobj)
            seg = np.transpose(seg, (2, 1, 0))
            seg[seg > 1] = 1  ## only keep label one
            seg_list.append(np.expand_dims(seg,axis=0))

        imgs = np.concatenate(img_list, axis=0)
        imgs = torch.from_numpy(imgs)
        imgs = imgs.to(device=device, dtype=torch.float32)
        segs = np.concatenate(seg_list, axis=0)
        segs = torch.from_numpy(segs)
        segs = segs.to(device=device, dtype=torch.float32)
        input_imgs = imgs[0:2,:].unsqueeze(1)
        input_segs = segs[0:2,:].unsqueeze(1)
        input_data = torch.concat([input_imgs,input_segs], dim=1)
        input_data = input_data.unsqueeze(0)
        
        time_interval = study_dates[1:3] - study_dates[0:2]
        print(study_dates, time_interval)
        time_interval = torch.from_numpy(time_interval)
        time_interval = time_interval.to(device=device, dtype=torch.float32)
        time_interval = time_interval.unsqueeze(0)
        scan_save_dir = os.path.join(save_dir, patient+'_'+scans[0])
        if not os.path.isdir(scan_save_dir):
            os.mkdir(scan_save_dir)
        with torch.no_grad():
            pred_img, pred_seg, state = Generator(input_data, time_interval)
            #print(pred_seg.shape)
            for j in range(pred_seg.shape[1]):
                mask = pred_seg[:,j].squeeze()
                mask = torch.sigmoid(mask)
                mask = (mask>0.5).cpu().numpy()
                mask = np.transpose(mask, (2,1,0)).astype(np.int32)
                largestCC = getlargestcc(mask)
                mask = mask * largestCC

                img = pred_img[:,j].squeeze()
                img = img.cpu().numpy()
                img = np.transpose(img, (2,1,0))
                
                img_to_save = nib.Nifti1Image(img, img_affine)
                mask_to_save = nib.Nifti1Image(mask, img_affine)
                
                save_name = os.path.join(scan_save_dir, patient + "_" + scans[j+1]+".nii.gz")
                nib.save(img_to_save,save_name)
            
                mask_save_name = os.path.join(scan_save_dir, patient + "_" + scans[j+1]+"_seg.nii.gz")
                nib.save(mask_to_save,mask_save_name)
