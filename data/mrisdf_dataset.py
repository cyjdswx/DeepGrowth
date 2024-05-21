import itertools
from math import gamma
import os
import numpy as np
from torch.nn.modules import transformer
from torchvision import transforms
from einops import rearrange, repeat
import torch
from torch.utils import data
from torch.utils.data import Dataset
import nibabel as nib
#import torchvision.transforms.functional as F
import torch.nn.functional as F
from datetime import datetime
from itertools import combinations
import random
class randomFilp_points(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, pnts, sample_coord):
        #dims_to_flip = []
        if torch.rand(1) < self.p:
            pnts[:,0] = -pnts[:,0]
            sample_coord[:,0] = -sample_coord[:,0]
        if torch.rand(1) < self.p:
            pnts[:,1] = -pnts[:,1]
            sample_coord[:,1] = -sample_coord[:,1]
        if torch.rand(1) < self.p:
            pnts[:,2] = -pnts[:,2]
            sample_coord[:,2] = -sample_coord[:,2]
        return pnts, sample_coord

class randomFlipEX(object):
    #### randomly filp points cloud and image tensor
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, pnts, sample_coord, tensor):
        c, d, h, w = tensor.shape
        dims_to_flip = []
        if torch.rand(1) < self.p:
            pnts[:,0] = -pnts[:,0]
            sample_coord[:,0] = -sample_coord[:,0]
            dims_to_flip.append(1)
        if torch.rand(1) < self.p:
            pnts[:,1] = -pnts[:,1]
            sample_coord[:,1] = -sample_coord[:,1]
            dims_to_flip.append(2)
        if torch.rand(1) < self.p:
            pnts[:,2] = -pnts[:,2]
            sample_coord[:,2] = -sample_coord[:,2]
            dims_to_flip.append(3)
        tensor = torch.flip(tensor, dims_to_flip)
        return pnts, sample_coord, tensor

class randompcShift(object):
    def __init__(self,  normalized=True, shape=None):
        self.normalized = normalized
        
    @staticmethod
    def get_boundingbox(coords):
        #### get the bounding box of the SDF
        min_d, max_d = coords[:, 0].min().item(), coords[:, 0].max().item()
        min_h, max_h = coords[:, 1].min().item(), coords[:, 1].max().item()
        min_w, max_w = coords[:, 2].min().item(), coords[:, 2].max().item()
        return min_d, max_d, min_h, max_h, min_w, max_w


    def __call__(self, pnts, sample_coord):
        """
        Randomly shift a 3D tensor along each axis.

        :param tensor: The input tensor of shape (C, D, H, W), where C is the number of channels,
                   D is the depth, H is the height, and W is the width.
        :param max_shift: A tuple of three integers, indicating the maximum shift in each dimension (D, H, W).
        :return: A shifted tensor of the same shape as the input.
        """
        #if
        #d, h, w = seg.shape
        min_d, max_d, min_h, max_h, min_w, max_w = self.get_boundingbox(pnts)
        # Randomly choose the shift distances for each axis
        shift_max = 0.2 * (1.0 - max_d)
        shift_min = 0.2 * (-1.0 - min_d)
        d_shift = (shift_min - shift_max) * torch.rand((1,)).item() + shift_max
        shift_max = 0.2 * (1.0 - max_h)
        shift_min = 0.2 * (-1.0 - min_h)
        h_shift = (shift_min - shift_max) * torch.rand((1,)).item() + shift_max
        shift_max = 0.2 * (1.0 - max_w)
        shift_min = 0.2 * (-1.0 - min_w)
        w_shift = (shift_min - shift_max) * torch.rand((1,)).item() + shift_max
        offset = np.asarray([d_shift, h_shift, w_shift])
        offset = offset.reshape(1,3)
        n,_ = pnts.shape
        shifted_pnts = pnts + np.tile(offset,(n,1))
        n,_ = sample_coord.shape
        shifted_sample_coord = sample_coord + np.tile(offset,(n,1))
        #out_index = (coords > 79).any(axis=1) | (coords < 0).any(axis=1)
        #if torch.any(out_index):
        #    assert ValueError
        #    print( d_shift, h_shift, w_shift)
        #coords = coords / 79 * 2 - 1 
        return shifted_pnts, shifted_sample_coord

class RandomNoise(object):
    def __init__(self, mean=0, std=0.02):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise

class randomShiftNeo(object):
    def __init__(self,  patch_size, normalized=True, shape=None):
        self.patch_size = patch_size
        self.normalized = normalized
        self.shift_factor = 0.2

    @staticmethod
    def get_boundingbox(coords):
        #### get the bounding box of the SDF
        min_d, max_d = coords[:, 0].min().item(), coords[:, 0].max().item()
        min_h, max_h = coords[:, 1].min().item(), coords[:, 1].max().item()
        min_w, max_w = coords[:, 2].min().item(), coords[:, 2].max().item()
        return min_d, max_d, min_h, max_h, min_w, max_w

    def __call__(self, pnts, sample_coord, tensor):
        """
        Randomly shift a 3D tensor along each axis.

        :param tensor: The input tensor of shape (C, D, H, W), where C is the number of channels,
                   D is the depth, H is the height, and W is the width.
        :param max_shift: A tuple of three integers, indicating the maximum shift in each dimension (D, H, W).
        :return: A shifted tensor of the same shape as the input.
        """
        d, h, w = self.patch_size
        #min_d, max_d, min_h, max_h, min_w, max_w = self.get_boundingbox(pnts)

        # Randomly choose the shift distances for each axis
        shift_max = int(self.shift_factor * d)
        shift_min = -1 * int(self.shift_factor * d)
        #d_shift = (shift_min - shift_max) * torch.rand((1,)).item() + shift_max
        d_shift = torch.randint(shift_min, shift_max, (1,)).item()
        #d_pnt_shift = float(d_shift) / 79 * 2
        d_pnt_shift = float(d_shift) / (d - 1) * 2

        shift_max = int(self.shift_factor * h)
        shift_min = -1 * int(self.shift_factor * h)
        #h_shift = (shift_min - shift_max) * torch.rand((1,)).item() + shift_max
        #print(shift_min,shift_max)
        h_shift = torch.randint(shift_min, shift_max, (1,)).item()
        #h_pnt_shift = float(h_shift) / 79 * 2
        h_pnt_shift = float(h_shift) / (h - 1) * 2

        shift_max = int(self.shift_factor * w)
        shift_min = -1 * int(self.shift_factor * w)
        #w_shift = (shift_min - shift_max) * torch.rand((1,)).item() + shift_max
        w_shift = torch.randint(shift_min, shift_max, (1,)).item()
        #w_pnt_shift = float(w_shift) / 79 * 2
        w_pnt_shift = float(w_shift) / (w - 1) * 2
        
        ### points
        pnt_offset = np.asarray([d_pnt_shift, h_pnt_shift, w_pnt_shift])
        pnt_offset = pnt_offset.reshape(1,3)
        n,_ = pnts.shape
        shifted_pnts = pnts + np.tile(pnt_offset,(n,1))
        n,_ = sample_coord.shape
        shifted_sample_coord = sample_coord + np.tile(pnt_offset,(n,1))
        
        ######### img
        #pd = [max(-1 * w_shift,0),max(w_shift,0),max(-1 * h_shift,0),max(h_shift,0),max(-1 * d_shift,0),max(d_shift,0)]
        pd = [max(w_shift,0),max(-1 * w_shift,0),max(h_shift,0),max(-1 * h_shift,0), max(d_shift,0),max(-1 * d_shift,0)]
        _, d, h, w = tensor.shape
        # Create an affine transformation matrix for the random shift
        padded_img = F.pad(tensor, pd, 'constant', 0)
        # Create an identity grid and apply the affine transformation
        starts = [pd[5], pd[3], pd[1]]
        ends = [starts[0] + d, starts[1] + h, starts[2] + w]
        shifted_img = padded_img[:, starts[0]:ends[0],starts[1]:ends[1], starts[2]:ends[2]]
        #print(d_shift,h_shift,w_shift)
        return shifted_pnts, shifted_sample_coord, shifted_img

class randomShiftEX(object):
    def __init__(self,  patch_size, normalized=True, shape=None):
        self.patch_size = patch_size
        self.normalized = normalized
        self.shift_factor = 0.3

    @staticmethod
    def get_boundingbox(coords):
        #### get the bounding box of the SDF
        min_d, max_d = coords[:, 0].min().item(), coords[:, 0].max().item()
        min_h, max_h = coords[:, 1].min().item(), coords[:, 1].max().item()
        min_w, max_w = coords[:, 2].min().item(), coords[:, 2].max().item()
        return min_d, max_d, min_h, max_h, min_w, max_w

    def __call__(self, pnts, sample_coord, tensor):
        """
        Randomly shift a 3D tensor along each axis.

        :param tensor: The input tensor of shape (C, D, H, W), where C is the number of channels,
                   D is the depth, H is the height, and W is the width.
        :param max_shift: A tuple of three integers, indicating the maximum shift in each dimension (D, H, W).
        :return: A shifted tensor of the same shape as the input.
        """
        d, h, w = self.patch_size
        min_d, max_d, min_h, max_h, min_w, max_w = self.get_boundingbox(pnts)
        # Randomly choose the shift distances for each axis
        shift_max = int(self.shift_factor * (1.0 - max_d) / 2 * (d - 1))
        shift_min = int(self.shift_factor * (-1.0 - min_d) / 2 * (d - 1))
        #d_shift = (shift_min - shift_max) * torch.rand((1,)).item() + shift_max
        d_shift = torch.randint(shift_min, shift_max, (1,)).item()
        #d_pnt_shift = float(d_shift) / 79 * 2
        d_pnt_shift = float(d_shift) / (d - 1) * 2

        shift_max = int(self.shift_factor * (1.0 - max_h) / 2 * (h - 1))
        shift_min = int(self.shift_factor * (-1.0 - min_h) / 2 * (h - 1))
        #h_shift = (shift_min - shift_max) * torch.rand((1,)).item() + shift_max
        #print(shift_min,shift_max)
        h_shift = torch.randint(shift_min, shift_max, (1,)).item()
        #h_pnt_shift = float(h_shift) / 79 * 2
        h_pnt_shift = float(h_shift) / (h - 1) * 2

        shift_max = int(self.shift_factor * (1.0 - max_w) / 2 * (w - 1))
        shift_min = int(self.shift_factor * (-1.0 - min_w) / 2 * (w - 1))
        #w_shift = (shift_min - shift_max) * torch.rand((1,)).item() + shift_max
        w_shift = torch.randint(shift_min, shift_max, (1,)).item()
        #w_pnt_shift = float(w_shift) / 79 * 2
        w_pnt_shift = float(w_shift) / (w - 1) * 2

        pnt_offset = np.asarray([d_pnt_shift, h_pnt_shift, w_pnt_shift])
        pnt_offset = pnt_offset.reshape(1,3)
        n,_ = pnts.shape
        shifted_pnts = pnts + np.tile(pnt_offset,(n,1))
        n,_ = sample_coord.shape
        shifted_sample_coord = sample_coord + np.tile(pnt_offset,(n,1))
        
        ######### img
        #pd = [max(-1 * w_shift,0),max(w_shift,0),max(-1 * h_shift,0),max(h_shift,0),max(-1 * d_shift,0),max(d_shift,0)]
        pd = [max(w_shift,0),max(-1 * w_shift,0),max(h_shift,0),max(-1 * h_shift,0), max(d_shift,0),max(-1 * d_shift,0)]
        _, d, h, w = tensor.shape
        # Create an affine transformation matrix for the random shift
        padded_img = F.pad(tensor, pd, 'constant', 0)
        # Create an identity grid and apply the affine transformation
        starts = [pd[5], pd[3], pd[1]]
        ends = [starts[0] + d, starts[1] + h, starts[2] + w]
        shifted_img = padded_img[:, starts[0]:ends[0],starts[1]:ends[1], starts[2]:ends[2]]
        #print(d_shift,h_shift,w_shift)
        return shifted_pnts, shifted_sample_coord, shifted_img

class randomShift(object):
    def __init__(self,  pad_if_needed=True, fill=0.0, padding_mode='constant'):
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
    
    @staticmethod
    def get_boundingbox(sdf):
        #### get the bounding box of the SDF
        d, h, w = sdf.shape
        # Create a mask where the condition is true
        object_mask = sdf < 1
        object_indices = object_mask.nonzero()
        if object_indices.size(0) == 0: ##### no values
            return 0, d - 1, 0, h - 1, 0, w - 1
        min_d, max_d = object_indices[:, 0].min().item(), object_indices[:, 0].max().item()
        min_h, max_h = object_indices[:, 1].min().item(), object_indices[:, 1].max().item()
        min_w, max_w = object_indices[:, 2].min().item(), object_indices[:, 2].max().item()
        
        return min_d, max_d, min_h, max_h, min_w, max_w

    def __call__(self, tensor):
        """
        Randomly shift a 3D tensor along each axis.

        :param tensor: The input tensor of shape (C, D, H, W), where C is the number of channels,
                   D is the depth, H is the height, and W is the width.
        :param max_shift: A tuple of three integers, indicating the maximum shift in each dimension (D, H, W).
        :return: A shifted tensor of the same shape as the input.
        """
        c, d, h, w = tensor.shape
        img = tensor[0,:]
        seg = tensor[1,:]
        sdf = tensor[2,:]
        min_d, max_d, min_h, max_h, min_w, max_w = self.get_boundingbox(sdf)
        # Randomly choose the shift distances for each axis
        #shifts = [torch.randint(-shift, shift + 1, (1,)).item() for shift in max_shift]
        d_shift = torch.randint(-1 * min_d, d - max_d, (1,)).item()
        h_shift = torch.randint(-1 * min_h, h - max_h, (1,)).item()
        w_shift = torch.randint(-1 * min_w, w - max_w, (1,)).item()
        pd = [max(-1 * w_shift,0),max(w_shift,0),max(-1 * h_shift,0),max(h_shift,0),max(-1 * d_shift,0),max(d_shift,0)]
        #print(d_shift, h_shift, w_shift, pd)
        # Create an affine transformation matrix for the random shift
        padded_img = F.pad(img, pd, 'constant', 0)
        padded_seg = F.pad(seg, pd, 'constant', 0)
        padded_sdf = F.pad(sdf, pd, 'constant', 1)
        # Create an identity grid and apply the affine transformation
        starts = [pd[4] + d_shift, pd[2] + h_shift, pd[0] + w_shift]
        ends = [pd[4] + d_shift + d, pd[2] + h_shift + h, pd[0] + w_shift + w]
        shifted_img = padded_img[starts[0]:ends[0],starts[1]:ends[1], starts[2]:ends[2]]
        shifted_seg = padded_seg[starts[0]:ends[0],starts[1]:ends[1], starts[2]:ends[2]]
        shifted_sdf = padded_sdf[starts[0]:ends[0],starts[1]:ends[1], starts[2]:ends[2]]
        return shifted_img, shifted_seg, shifted_sdf

class LongMRI(object):
    def __init__(self, root_path, scans):
        assert isinstance(scans, list),\
                "Class MultiMRI2d needs list of inputfiles"
        self.inputfiles = scans
        self.images_handle = []
        self.segs_handle = []
        self.study_dates = []
        self.patient_id = []
        self.maximal_period = 96
        scans.sort()
        try:
            baseline_date = datetime.strptime(scans[0].split('_')[1], '%Y%m%d')
        except ValueError: 
            print('invalid date',scans[0])
            raise ValueError('invalid', root_path, scans[0])

        for scan in scans:
            ### time internal in months
            study_date = scan.split('_')[1]
            try:
                study_date = datetime.strptime(study_date, '%Y%m%d')
            except ValueError:
                raise ValueError('invalid', root_path, study_date)
            day_diff = study_date.day - baseline_date.day
            month_diff = study_date.month - baseline_date.month
            year_diff = study_date.year - baseline_date.year
            time_diff = year_diff * 12 + month_diff + float(day_diff) / 30
            if time_diff > self.maximal_period - 1:
                time_diff = self.maximal_period - 1
            self.study_dates.append(time_diff)
            
            ### images
            img_path = os.path.join(root_path, scan + '.nii.gz')
            assert os.path.isfile(img_path), "missing file"
            image_handle = nib.load(img_path)
            self.images_handle.append(image_handle)
            
            ### segs
            seg_path = os.path.join(root_path, scan + '_seg.nii.gz') 
            assert os.path.isfile(seg_path), "missing segmentation"
            seg_handle = nib.load(seg_path)
            self.segs_handle.append(seg_handle)

        self.patient_id.append(scans[0].split('_')[0])
            
    def get_MRI_shape(self):
        shapes = self.images_handle[0].header.get_data_shape()
        return shapes
   
    def get_num_series(self):
        return len(self.images_handle)

    def get_patient_id(self):
        return self.patient_id

    def get_img_data_tp(self, tp):
        assert tp >= 0 and tp < len(self.images_handle),\
                "incorrect index"
        #print(self.images_handle[tp])
        image_data =  np.asarray(self.images_handle[tp].dataobj)
        seg_data =  np.asarray(self.segs_handle[tp].dataobj)

        study_date = self.study_dates[tp]
        return image_data, seg_data, study_date

class LongMRIDiffSdf(object):
    def __init__(self, root_path, scans):
        assert isinstance(scans, list),\
                "Class MultiMRI2d needs list of inputfiles"
        self.inputfiles = scans
        self.images_handle = []
        self.segs_handle = []
        self.sdfs_handle = []
        self.densesdf_handle = []
        self.gridsdf_handle = []
        self.pc_handle = []
        self.study_dates = []
        self.patient_id = []
        self.maximal_period = 96
        scans.sort()
        try:
            baseline_date = datetime.strptime(scans[0].split('_')[1], '%Y%m%d')
        except ValueError: 
            print('invalid date',scans[0])
            raise ValueError('invalid', root_path, scans[0])

        for scan in scans:
            ### time internal in months
            study_date = scan.split('_')[1]
            try:
                study_date = datetime.strptime(study_date, '%Y%m%d')
            except ValueError:
                raise ValueError('invalid', root_path, study_date)
            day_diff = study_date.day - baseline_date.day
            month_diff = study_date.month - baseline_date.month
            year_diff = study_date.year - baseline_date.year
            time_diff = year_diff * 12 + month_diff + float(day_diff) / 30
            if time_diff > self.maximal_period - 1:
                time_diff = self.maximal_period - 1
            self.study_dates.append(time_diff)
            
            ### images
            img_path = os.path.join(root_path, scan + '.nii.gz')
            assert os.path.isfile(img_path), "missing file"
            image_handle = nib.load(img_path)
            self.images_handle.append(image_handle)
            
            ### segs
            seg_path = os.path.join(root_path, scan + '_seg.nii.gz') 
            assert os.path.isfile(seg_path), "missing segmentation"
            seg_handle = nib.load(seg_path)
            self.segs_handle.append(seg_handle)
            
            ### densesdfs
            dense_sdf_path = os.path.join(root_path, scan + '_densesdf.npy')
            assert os.path.isfile(dense_sdf_path), "missing dense signed distance function"
            self.densesdf_handle.append(dense_sdf_path)
            
            ### point cloud
            #pc_path = os.path.join(root_path, scan + '_smoothedls.npy')
            gridsdf_path = os.path.join(root_path, scan + '_gridsdf.npy')
            assert os.path.isfile(gridsdf_path), "missing grid sdf"
            self.gridsdf_handle.append(gridsdf_path)
            
            ### point cloud
            #pc_path = os.path.join(root_path, scan + '_smoothedls.npy')
            pc_path = os.path.join(root_path, scan + '_zls.npy')
            assert os.path.isfile(pc_path), "missing point cloud"
            self.pc_handle.append(pc_path)

        self.patient_id.append(scans[0].split('_')[0])
            
    def get_MRI_shape(self):
        shapes = self.images_handle[0].header.get_data_shape()
        return shapes
   
    def get_num_series(self):
        return len(self.images_handle)

    def get_patient_id(self):
        return self.patient_id

    def get_img_data_tp(self, tp):
        assert tp >= 0 and tp < len(self.images_handle),\
                "incorrect index"
        #print(self.images_handle[tp])
        image_data =  np.asarray(self.images_handle[tp].dataobj)
        seg_data =  np.asarray(self.segs_handle[tp].dataobj)
        densesdf_data = np.load(self.densesdf_handle[tp])
        gridsdf_data = np.load(self.gridsdf_handle[tp])
        pc_data = np.load(self.pc_handle[tp])

        study_date = self.study_dates[tp]
        return image_data, seg_data, densesdf_data, gridsdf_data, pc_data,study_date

class LongMRIDenseSdf(object):
    def __init__(self, root_path, scans):
        assert isinstance(scans, list),\
                "Class MultiMRI2d needs list of inputfiles"
        self.inputfiles = scans
        self.images_handle = []
        self.segs_handle = []
        self.sdfs_handle = []
        self.densesdf_handle = []
        self.pc_handle = []
        self.study_dates = []
        self.patient_id = []
        self.maximal_period = 96
        scans.sort()
        try:
            baseline_date = datetime.strptime(scans[0].split('_')[1], '%Y%m%d')
        except ValueError: 
            print('invalid date',scans[0])
            raise ValueError('invalid', root_path, scans[0])

        for scan in scans:
            ### time internal in months
            study_date = scan.split('_')[1]
            try:
                study_date = datetime.strptime(study_date, '%Y%m%d')
            except ValueError:
                raise ValueError('invalid', root_path, study_date)
            day_diff = study_date.day - baseline_date.day
            month_diff = study_date.month - baseline_date.month
            year_diff = study_date.year - baseline_date.year
            time_diff = year_diff * 12 + month_diff + float(day_diff) / 30
            if time_diff > self.maximal_period - 1:
                time_diff = self.maximal_period - 1
            self.study_dates.append(time_diff)
            
            ### images
            img_path = os.path.join(root_path, scan + '.nii.gz')
            assert os.path.isfile(img_path), "missing file"
            image_handle = nib.load(img_path)
            self.images_handle.append(image_handle)
            
            ### segs
            seg_path = os.path.join(root_path, scan + '_seg.nii.gz') 
            assert os.path.isfile(seg_path), "missing segmentation"
            seg_handle = nib.load(seg_path)
            self.segs_handle.append(seg_handle)
            
            ### densesdfs
            dense_sdf_path = os.path.join(root_path, scan + '_densesdf.npy')
            assert os.path.isfile(dense_sdf_path), "missing dense signed distance function"
            self.densesdf_handle.append(dense_sdf_path)
            
            ### point cloud
            #pc_path = os.path.join(root_path, scan + '_smoothedls.npy')
            pc_path = os.path.join(root_path, scan + '_zls.npy')
            assert os.path.isfile(pc_path), "missing point cloud"
            self.pc_handle.append(pc_path)

        self.patient_id.append(scans[0].split('_')[0])
            
    def get_MRI_shape(self):
        shapes = self.images_handle[0].header.get_data_shape()
        return shapes
   
    def get_num_series(self):
        return len(self.images_handle)

    def get_patient_id(self):
        return self.patient_id

    def get_img_data_tp(self, tp):
        assert tp >= 0 and tp < len(self.images_handle),\
                "incorrect index"
        #print(self.images_handle[tp])
        image_data =  np.asarray(self.images_handle[tp].dataobj)
        seg_data =  np.asarray(self.segs_handle[tp].dataobj)
        densesdf_data = np.load(self.densesdf_handle[tp])
        pc_data = np.load(self.pc_handle[tp])

        study_date = self.study_dates[tp]
        return image_data, seg_data, densesdf_data, pc_data, study_date

class LongMriFullDatasetThreeDA(Dataset):
    ##### use first two scans as input and pick one of the latter scan as output
    def __init__(self, dataroot, filelist, patch_size, is_train) -> None:
        super(LongMriFullDatasetThreeDA, self).__init__()
        self.dataroot = dataroot
        self.patient_handles = []
        self.scan_list = []
        self.filelist = filelist
        self.max_num_series = 10
        self.maximal_period = 96
        self.input_num = 3
        ## data augumentation
        self.shift = randomShiftEX()
        self.flip = randomFlipEX()
        self.transform = transforms.Compose([
                                    RandomNoise(),
                                    transforms.ColorJitter(brightness=0.3, contrast=0.3) # Random contrast adjustment
                                    ])
        self.istrain = is_train
        with open(os.path.join(self.dataroot, self.filelist)) as f:
            self.patientlist = f.read().splitlines()
        self.__load_images()
        self.__get_indexes()
        
    def __load_images(self):
        for patient in self.patientlist:
            #scan_list = []
            patient_dir = os.path.join(self.dataroot, patient)
            scans = os.listdir(patient_dir)
            scans = [scan.split('.')[0] for scan in scans if scan.endswith('.nii.gz') and not scan.endswith('_seg.nii.gz') and not scan.endswith('_sdf.nii.gz')]
            scans.sort()
            print(patient, scans)
            MRI = LongMRIDenseSdf(patient_dir, scans)
            self.patient_handles.append(MRI)
    
    def __get_indexes(self):
        for handle in self.patient_handles:
            num_series = handle.get_num_series()
            sample_index = (handle,0,1,2) 
            self.scan_list.append(sample_index)

    def __getitem__(self, index):
        patient_handle = self.scan_list[index][0]
        num_series = patient_handle.get_num_series()
        assert num_series <= self.max_num_series, "We only accept patient with atmost 10 scans"
        img_list = []
        pnts_list = []
        normals_list = []
        segs_list = []
        sample_coord_list = []
        sample_sdf_list = []
        study_dates = []
        for i in range(self.input_num):
            scan_id = self.scan_list[index][i+1]
            image, seg, densesdf, p, study_date = patient_handle.get_img_data_tp(scan_id)
            image = np.transpose(image, (2, 1 ,0))
            img_list.append(np.expand_dims(image,axis=0))
            
            seg = np.transpose(seg, (2, 1, 0))
            seg[seg > 1] = 1  ## only keep one
            segs_list.append(np.expand_dims(seg, axis=0))
            
            pnts = p[:,0:3]
            pnts_list.append(np.expand_dims(pnts,axis=0))
            normals = p[:,3:6]
            normals_list.append(np.expand_dims(normals,axis=0))

            sample_coord = densesdf[:,0:3]
            sample_sdf = densesdf[:,3]
            sample_coord_list.append(np.expand_dims(sample_coord,axis=0))
            sample_sdf_list.append(np.expand_dims(sample_sdf,axis=0))
            
            study_dates.append(study_date)

        imgs = np.concatenate(img_list)
        imgs = torch.from_numpy(imgs)
        seg = np.concatenate(segs_list,axis=0)
        seg = torch.from_numpy(seg)
        
        s , d, h, w = seg.shape
        _,n_p, _ =  pnts_list[0].shape
        _,n_s, _ =  sample_coord_list[0].shape
        pnts = np.concatenate(pnts_list, axis=0)
        pnts = torch.from_numpy(pnts)
        normals = np.concatenate(normals_list, axis=0)
        normals = torch.from_numpy(normals)
        sample_coord = np.concatenate(sample_coord_list, axis=0)
        sample_coord = torch.from_numpy(sample_coord)
        sample_sdf = np.concatenate(sample_sdf_list, axis=0)
        sample_sdf = torch.from_numpy(sample_sdf)
        if self.istrain:
            data_tensor = torch.concat([imgs,seg],dim=0)
            pnts = rearrange(pnts,'s n c-> (s n) c')
            sample_coord = rearrange(sample_coord, 's n c -> (s n) c')
            pnts, sample_coord, data_tensor = self.flip(pnts, sample_coord, data_tensor)
            pnts, sample_coord, data_tensor = self.shift(pnts, sample_coord, data_tensor)
            imgs = data_tensor[:3,:]
            seg = data_tensor[3:,:]
            
            imgs = rearrange(imgs, 'c d h w->c d 1 h w')
            imgs = self.transform(imgs)
            imgs = rearrange(imgs, 'c d 1 h w->c d h w')
            
            sample_coord = rearrange(sample_coord,'(s n) c -> s n c',s=self.input_num, n=n_s) 
            pnts = rearrange(pnts,'(s n) c -> s n c',s=self.input_num,n=n_p)
            sample_coord_selected_list = []
            sample_sdf_selected_list = []
            for i in range(self.input_num):
                sample_coord_i = sample_coord[i]
                sample_sdf_i = sample_sdf[i]
                valid_index = torch.all(torch.logical_and(sample_coord_i <= 1,sample_coord_i >= -1), dim=1)
                sample_coord_i = sample_coord_i[valid_index]
                sample_sdf_i = sample_sdf_i[valid_index]
                sample_size = 50000
                random_indices = torch.randperm(sample_coord_i.shape[0])[:sample_size]
                sample_coord_i = sample_coord_i[random_indices]
                sample_sdf_i = sample_sdf_i[random_indices]
                sample_coord_selected_list.append(sample_coord_i.unsqueeze(0))
                sample_sdf_selected_list.append(sample_sdf_i.unsqueeze(0))
            sample_coord = torch.concat(sample_coord_selected_list,dim=0)
            sample_sdf = torch.concat(sample_sdf_selected_list,dim=0)
        
        xyz_coords = np.mgrid[:d, :h, :w].astype(np.float32)
        xyz_coords[0, ...] = xyz_coords[0, ...] / (d - 1)
        xyz_coords[1, ...] = xyz_coords[1, ...] / (h - 1)
        xyz_coords[2, ...] = xyz_coords[2, ...] / (w - 1)
        xyz_coords = (xyz_coords - 0.5) * 2
        xyz_coords = rearrange(xyz_coords, 'c d h w -> 1 (d h w) c')
        xyz_coords = repeat(xyz_coords, '1 d c -> s d c', s=s)
        grid_coord = torch.from_numpy(xyz_coords)
        grid_occ = rearrange(seg,'s d h w -> s (d h w)')
        
        sample_occ = torch.zeros_like(sample_sdf)
        sample_occ[sample_sdf < 0] = 1
        imgs = imgs * 2 - 1
        study_dates = np.array(study_dates)
        study_dates[study_dates > self.maximal_period] = self.maximal_period
        study_dates = (study_dates / self.maximal_period) * 2 - 1
        study_dates = torch.from_numpy(study_dates)
        input_dict = {'pnts':pnts,
                      'normals':normals, 
                      'seg': seg,
                      'img':imgs, 
                      'study_dates': study_dates, 
                      'sample_coord':sample_coord,
                      'sample_sdf': sample_sdf,
                      'sample_occ':sample_occ,
                      'grid_coord':grid_coord,
                      'grid_occ':grid_occ
                      }
        return input_dict
    
    def __len__(self):
        return len(self.scan_list)

class LongMriImageMoreDataset(Dataset):
    ##### use first two scans as input and pick one of the latter scan as output
    def __init__(self, dataroot, filelist, patch_size, is_train) -> None:
        super(LongMriImageMoreDataset, self).__init__()
        self.dataroot = dataroot
        self.patient_handles = []
        self.scan_list = []
        self.filelist = filelist
        self.max_num_series = 10
        self.maximal_period = 96
        self.input_num = 3
        ## data augumentation
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=0,translate=(0.2,0.2))])
        
        self.istrain = is_train
        with open(os.path.join(self.dataroot, self.filelist)) as f:
            self.patientlist = f.read().splitlines()
        self.__load_images()
        self.__get_indexes()
        
    def __load_images(self):
        for patient in self.patientlist:
            #scan_list = []
            patient_dir = os.path.join(self.dataroot, patient)
            scans = os.listdir(patient_dir)
            scans = [scan.split('.')[0] for scan in scans if scan.endswith('.nii.gz') and not scan.endswith('_seg.nii.gz') and not scan.endswith('_sdf.nii.gz')]
            scans.sort()
            print(patient, scans)
            #MRI = LongMRIDenseSdf(patient_dir, scans)
            MRI = LongMRI(patient_dir, scans)
            self.patient_handles.append(MRI)
    
    def __get_indexes(self):

        for handle in self.patient_handles:
            num_series = handle.get_num_series()
            combs = itertools.combinations(range(num_series),3)
            #for comb in combs:
            for i in range(num_series-2):
                sample_index = (handle,0,1,) + (i+2,)
                self.scan_list.append(sample_index)
                print(sample_index)

    def __getitem__(self, index):
        patient_handle = self.scan_list[index][0]
        num_series = patient_handle.get_num_series()
        assert num_series <= self.max_num_series, "We only accept patient with atmost 10 scans"
        img_list = []
        segs_list = []
        study_dates = []
        for i in range(self.input_num):
            scan_id = self.scan_list[index][i+1]
            image, seg, study_date = patient_handle.get_img_data_tp(scan_id)
            image = np.transpose(image, (2, 1 ,0))
            img_list.append(np.expand_dims(image,axis=0))
            
            seg = np.transpose(seg, (2, 1, 0))
            seg[seg > 1] = 1  ## only keep label one
            segs_list.append(np.expand_dims(seg, axis=0))

            study_dates.append(study_date)

        imgs = np.concatenate(img_list)
        imgs = torch.from_numpy(imgs)
        seg = np.concatenate(segs_list,axis=0)
        seg = torch.from_numpy(seg)
        if self.istrain:
            data_tensor = torch.concat([imgs,seg],dim=0)
            data_tensor = self.transform(data_tensor)
            imgs = data_tensor[:3,:]
            seg = data_tensor[3:,:]
            
        imgs = imgs * 2 - 1
        ## study_dates
        study_dates = np.array(study_dates)
        study_dates[study_dates > self.maximal_period] = self.maximal_period
        study_dates = (study_dates / self.maximal_period) * 2 - 1
        study_dates = torch.from_numpy(study_dates)
        input_dict = {'seg': seg,
                      'img':imgs, 
                      'study_dates': study_dates
                      }
        return input_dict
    
    def __len__(self):
        return len(self.scan_list)

class LongMriImageCompleteDataset(Dataset):
    ##### use first two scans as input and pick one of the latter scan as output
    def __init__(self, dataroot, filelist, patch_size, is_train) -> None:
        super(LongMriImageCompleteDataset, self).__init__()
        self.dataroot = dataroot
        self.patient_handles = []
        self.scan_list = []
        self.filelist = filelist
        self.max_num_series = 10
        self.maximal_period = 96
        self.input_num = 3
        ## data augumentation
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=0,translate=(0.2,0.2))])
        
        self.istrain = is_train
        with open(os.path.join(self.dataroot, self.filelist)) as f:
            self.patientlist = f.read().splitlines()
        self.__load_images()
        self.__get_indexes()
        
    def __load_images(self):
        for patient in self.patientlist:
            #scan_list = []
            patient_dir = os.path.join(self.dataroot, patient)
            scans = os.listdir(patient_dir)
            scans = [scan.split('.')[0] for scan in scans if scan.endswith('.nii.gz') and not scan.endswith('_seg.nii.gz') and not scan.endswith('_sdf.nii.gz')]
            scans.sort()
            print(patient, scans)
            #MRI = LongMRIDenseSdf(patient_dir, scans)
            MRI = LongMRI(patient_dir, scans)
            self.patient_handles.append(MRI)
    
    def __get_indexes(self):

        for handle in self.patient_handles:
            num_series = handle.get_num_series()
            combs = itertools.combinations(range(num_series),3)
            for comb in combs:
                sample_index = (handle,) + comb
                self.scan_list.append(sample_index)

    def __getitem__(self, index):
        patient_handle = self.scan_list[index][0]
        num_series = patient_handle.get_num_series()
        assert num_series <= self.max_num_series, "We only accept patient with atmost 10 scans"
        img_list = []
        segs_list = []
        study_dates = []
        for i in range(self.input_num):
            scan_id = self.scan_list[index][i+1]
            image, seg, study_date = patient_handle.get_img_data_tp(scan_id)
            image = np.transpose(image, (2, 1 ,0))
            img_list.append(np.expand_dims(image,axis=0))
            
            seg = np.transpose(seg, (2, 1, 0))
            seg[seg > 1] = 1  ## only keep label one
            segs_list.append(np.expand_dims(seg, axis=0))

            study_dates.append(study_date)

        imgs = np.concatenate(img_list)
        imgs = torch.from_numpy(imgs)
        seg = np.concatenate(segs_list,axis=0)
        seg = torch.from_numpy(seg)
        if self.istrain:
            data_tensor = torch.concat([imgs,seg],dim=0)
            data_tensor = self.transform(data_tensor)
            imgs = data_tensor[:3,:]
            seg = data_tensor[3:,:]
            
        imgs = imgs * 2 - 1
        ## study_dates
        study_dates = np.array(study_dates)
        study_dates[study_dates > self.maximal_period] = self.maximal_period
        study_dates = (study_dates / self.maximal_period) * 2 - 1
        study_dates = torch.from_numpy(study_dates)
        input_dict = {'seg': seg,
                      'img':imgs, 
                      'study_dates': study_dates
                      }
        return input_dict
    
    def __len__(self):
        return len(self.scan_list)

class LongMriImageClipDataset(Dataset):
    ##### use first two scans as input and pick one of the latter scan as output, randomly select 5 slides to train st-convlstm
    def __init__(self, dataroot, filelist, patch_size, is_train) -> None:
        super(LongMriImageClipDataset, self).__init__()
        self.dataroot = dataroot
        self.patient_handles = []
        self.scan_list = []
        self.filelist = filelist
        self.max_num_series = 10
        self.maximal_period = 96
        self.input_num = 3
        ## data augumentation
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=0,translate=(0.2,0.2))])
        
        self.istrain = is_train
        with open(os.path.join(self.dataroot, self.filelist)) as f:
            self.patientlist = f.read().splitlines()
        self.__load_images()
        self.__get_indexes()
        
    def __load_images(self):
        for patient in self.patientlist:
            #scan_list = []
            patient_dir = os.path.join(self.dataroot, patient)
            scans = os.listdir(patient_dir)
            scans = [scan.split('.')[0] for scan in scans if scan.endswith('.nii.gz') and not scan.endswith('_seg.nii.gz') and not scan.endswith('_sdf.nii.gz')]
            scans.sort()
            print(patient, scans)
            #MRI = LongMRIDenseSdf(patient_dir, scans)
            MRI = LongMRI(patient_dir, scans)
            self.patient_handles.append(MRI)
    
    def __get_indexes(self):
        for handle in self.patient_handles:
            num_series = handle.get_num_series()
            sample_index = (handle,0,1,2) 
            self.scan_list.append(sample_index)

    def __getitem__(self, index):
        patient_handle = self.scan_list[index][0]
        num_series = patient_handle.get_num_series()
        assert num_series <= self.max_num_series, "We only accept patient with atmost 10 scans"
        img_list = []
        segs_list = []
        study_dates = []
        for i in range(self.input_num):
            scan_id = self.scan_list[index][i+1]
            image, seg, study_date = patient_handle.get_img_data_tp(scan_id)
            image = np.transpose(image, (2, 1 ,0))
            img_list.append(np.expand_dims(image,axis=0))
            
            seg = np.transpose(seg, (2, 1, 0))
            seg[seg > 1] = 1  ## only keep label one
            segs_list.append(np.expand_dims(seg, axis=0))

            study_dates.append(study_date)
            
        imgs = np.concatenate(img_list)
        imgs = torch.from_numpy(imgs)
        seg = np.concatenate(segs_list,axis=0)
        seg = torch.from_numpy(seg)
        if self.istrain:
            data_tensor = torch.concat([imgs,seg],dim=0)
            data_tensor = self.transform(data_tensor)
            imgs = data_tensor[:3,:]
            seg = data_tensor[3:,:]
        imgs = imgs * 2 - 1
        s, d ,h ,w = imgs.shape
        start_slice = random.randint(0,d-5)
        imgs = imgs[:,start_slice:start_slice + 5,:]
        seg = seg[:,start_slice:start_slice + 5,:]
        ## study_dates
        study_dates = np.array(study_dates)
        study_dates[study_dates > self.maximal_period] = self.maximal_period
        study_dates = (study_dates / self.maximal_period) * 2 - 1
        study_dates = torch.from_numpy(study_dates)
        input_dict = {'seg': seg,
                      'img':imgs, 
                      'study_dates': study_dates
                      }
        return input_dict
    
    def __len__(self):
        return len(self.scan_list)

class LongMriImageDataset(Dataset):
    ##### use first two scans as input and pick one of the latter scan as output
    def __init__(self, dataroot, filelist, patch_size, is_train) -> None:
        super(LongMriImageDataset, self).__init__()
        self.dataroot = dataroot
        self.patient_handles = []
        self.scan_list = []
        self.filelist = filelist
        self.max_num_series = 10
        self.maximal_period = 96
        self.input_num = 3
        ## data augumentation
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=0,translate=(0.2,0.2))])
        
        self.istrain = is_train
        with open(os.path.join(self.dataroot, self.filelist)) as f:
            self.patientlist = f.read().splitlines()
        self.__load_images()
        self.__get_indexes()
        
    def __load_images(self):
        for patient in self.patientlist:
            #scan_list = []
            patient_dir = os.path.join(self.dataroot, patient)
            scans = os.listdir(patient_dir)
            scans = [scan.split('.')[0] for scan in scans if scan.endswith('.nii.gz') and not scan.endswith('_seg.nii.gz') and not scan.endswith('_sdf.nii.gz')]
            scans.sort()
            print(patient, scans)
            #MRI = LongMRIDenseSdf(patient_dir, scans)
            MRI = LongMRI(patient_dir, scans)
            self.patient_handles.append(MRI)
    
    def __get_indexes(self):
        for handle in self.patient_handles:
            num_series = handle.get_num_series()
            sample_index = (handle,0,1,2) 
            self.scan_list.append(sample_index)

    def __getitem__(self, index):
        patient_handle = self.scan_list[index][0]
        num_series = patient_handle.get_num_series()
        assert num_series <= self.max_num_series, "We only accept patient with atmost 10 scans"
        img_list = []
        segs_list = []
        study_dates = []
        for i in range(self.input_num):
            scan_id = self.scan_list[index][i+1]
            image, seg, study_date = patient_handle.get_img_data_tp(scan_id)
            image = np.transpose(image, (2, 1 ,0))
            img_list.append(np.expand_dims(image,axis=0))
            
            seg = np.transpose(seg, (2, 1, 0))
            seg[seg > 1] = 1  ## only keep label one
            segs_list.append(np.expand_dims(seg, axis=0))

            study_dates.append(study_date)

        imgs = np.concatenate(img_list)
        imgs = torch.from_numpy(imgs)
        seg = np.concatenate(segs_list,axis=0)
        seg = torch.from_numpy(seg)
        if self.istrain:
            data_tensor = torch.concat([imgs,seg],dim=0)
            data_tensor = self.transform(data_tensor)
            imgs = data_tensor[:3,:]
            seg = data_tensor[3:,:]
            
        imgs = imgs * 2 - 1
        ## study_dates
        study_dates = np.array(study_dates)
        study_dates[study_dates > self.maximal_period] = self.maximal_period
        study_dates = (study_dates / self.maximal_period) * 2 - 1
        study_dates = torch.from_numpy(study_dates)
        input_dict = {'seg': seg,
                      'img':imgs, 
                      'study_dates': study_dates
                      }
        return input_dict
    
    def __len__(self):
        return len(self.scan_list)

class LongMriFullDatasetDiff(Dataset):
    ##### use first two scans as input and pick one of the latter scan as output
    def __init__(self, dataroot, filelist, patch_size, is_train) -> None:
        super(LongMriFullDatasetDiff, self).__init__()
        self.dataroot = dataroot
        self.patient_handles = []
        self.scan_list = []
        self.filelist = filelist
        self.max_num_series = 10
        self.maximal_period = 96
        self.input_num = 3
        self.patch_size = patch_size
        ## data augumentation
        self.shift = randomShiftNeo(self.patch_size)
        self.flip = randomFlipEX()
        self.istrain = is_train
        with open(os.path.join(self.dataroot, self.filelist)) as f:
            self.patientlist = f.read().splitlines()
        self.__load_images()
        self.__get_indexes()
        
    def __load_images(self):
        for patient in self.patientlist:
            #scan_list = []
            patient_dir = os.path.join(self.dataroot, patient)
            scans = os.listdir(patient_dir)
            scans = [scan.split('.')[0] for scan in scans if scan.endswith('.nii.gz') and not scan.endswith('_seg.nii.gz') and not scan.endswith('_sdf.nii.gz')]
            scans.sort()
            print(patient, scans)
            MRI = LongMRIDiffSdf(patient_dir, scans)
            self.patient_handles.append(MRI)
    
    def __get_indexes(self):
        for handle in self.patient_handles:
            num_series = handle.get_num_series()
            sample_index = (handle,0,1,2) 
            self.scan_list.append(sample_index)

    def __getitem__(self, index):
        patient_handle = self.scan_list[index][0]
        num_series = patient_handle.get_num_series()
        assert num_series <= self.max_num_series, "We only accept patient with atmost 10 scans"
        img_list = []
        segs_list = []
        pnts_list = []
        normals_list = []
        sample_coord_list = []
        sample_sdf_list = []
        grid_coord_list = []
        grid_sdf_list = []
        study_dates = []
        for i in range(self.input_num):
            scan_id = self.scan_list[index][i+1]
            image, seg, densesdf, grid_sdf,p, study_date = patient_handle.get_img_data_tp(scan_id)
            image = np.transpose(image, (2, 1 ,0))
            img_list.append(np.expand_dims(image,axis=0))
            
            seg = np.transpose(seg, (2, 1, 0))
            seg[seg > 1] = 1  ## only keep one
            segs_list.append(np.expand_dims(seg, axis=0))
            
            pnts = p[:,0:3]
            pnts_list.append(np.expand_dims(pnts,axis=0))
            normals = p[:,3:6]
            normals_list.append(np.expand_dims(normals,axis=0))
            
            grid_coord = grid_sdf[:,0:3]
            grid_sdf = grid_sdf[:,3]
            grid_coord_list.append(np.expand_dims(grid_coord,axis=0))
            grid_sdf_list.append(np.expand_dims(grid_sdf,axis=0))

            sample_coord = densesdf[:,0:3]
            sample_sdf = densesdf[:,3]
            sample_coord_list.append(np.expand_dims(sample_coord,axis=0))
            sample_sdf_list.append(np.expand_dims(sample_sdf,axis=0))
            
            study_dates.append(study_date)

        imgs = np.concatenate(img_list)
        imgs = torch.from_numpy(imgs)
        seg = np.concatenate(segs_list,axis=0)
        seg = torch.from_numpy(seg)
        
        s , d, h, w = seg.shape
        _,n_p, _ =  pnts_list[0].shape
        _,n_s, _ =  sample_coord_list[0].shape
        pnts = np.concatenate(pnts_list, axis=0)
        pnts = torch.from_numpy(pnts)
        normals = np.concatenate(normals_list, axis=0)
        normals = torch.from_numpy(normals)
        sample_coord = np.concatenate(sample_coord_list, axis=0)
        sample_coord = torch.from_numpy(sample_coord)
        sample_sdf = np.concatenate(sample_sdf_list, axis=0)
        sample_sdf = torch.from_numpy(sample_sdf)
        
        grid_coord = np.concatenate(grid_coord_list, axis=0)
        grid_coord = torch.from_numpy(grid_coord)
        grid_sdf = np.concatenate(grid_sdf_list, axis=0)
        grid_sdf = torch.from_numpy(grid_sdf)
        if self.istrain:
            data_tensor = torch.concat([imgs,seg],dim=0)
            pnts = rearrange(pnts,'s n c-> (s n) c')
            sample_coord = rearrange(sample_coord, 's n c -> (s n) c')
            pnts, sample_coord, data_tensor = self.flip(pnts, sample_coord, data_tensor)
            pnts, sample_coord, data_tensor = self.shift(pnts, sample_coord, data_tensor)
            imgs = data_tensor[:3,:]
            seg = data_tensor[3:,:]
            
            sample_coord = rearrange(sample_coord,'(s n) c -> s n c',s=self.input_num, n=n_s) 
            pnts = rearrange(pnts,'(s n) c -> s n c',s=self.input_num,n=n_p)
            sample_coord_selected_list = []
            sample_sdf_selected_list = []
            
            sample_coord_i = sample_coord[0]
            valid_index = torch.all(torch.logical_and(sample_coord_i <= 1,sample_coord_i >= -1), dim=1)
            sample_coord_i = sample_coord_i[valid_index]
            sample_size = 50000
            random_indices = torch.randperm(sample_coord_i.shape[0])[:sample_size]
            while random_indices.shape[0] < sample_size:
                rest_sample = sample_size - random_indices.shape[0]
                random_indices_extra = torch.randperm(sample_coord_i.shape[0])[:rest_sample]
                random_indices = torch.concat([random_indices,random_indices_extra],dim=0)
            assert random_indices.shape[0]>=50000, "too few data"
            sample_coord_i = sample_coord_i[random_indices]
            for i in range(self.input_num):
                sample_sdf_i = sample_sdf[i]
                sample_sdf_i = sample_sdf_i[valid_index]
                sample_sdf_i = sample_sdf_i[random_indices]
                
                sample_coord_selected_list.append(sample_coord_i.unsqueeze(0))
                sample_sdf_selected_list.append(sample_sdf_i.unsqueeze(0))
            
            sample_coord = torch.concat(sample_coord_selected_list,dim=0)
            sample_sdf = torch.concat(sample_sdf_selected_list,dim=0)
        
        grid_occ = rearrange(seg,'s d h w -> s (d h w)')
        sample_occ = torch.zeros_like(sample_sdf)
        sample_occ[sample_sdf < 0] = 1
        
        imgs = imgs * 2 - 1
        study_dates = np.array(study_dates)
        study_dates[study_dates > self.maximal_period] = self.maximal_period
        study_dates = (study_dates / self.maximal_period) * 2 - 1
        study_dates = torch.from_numpy(study_dates)
        input_dict = {'pnts':pnts,
                      'normals':normals, 
                      'seg': seg,
                      'img':imgs, 
                      'study_dates': study_dates, 
                      'sample_coord':sample_coord,
                      'sample_sdf': sample_sdf,
                      'sample_occ':sample_occ,
                      'grid_coord':grid_coord,
                      'grid_sdf':grid_sdf,
                      'grid_occ':grid_occ
                      }
        return input_dict
    
    def __len__(self):
        return len(self.scan_list)

class LongMriFullDataset(Dataset):
    ##### use first two scans as input and pick one of the latter scan as output
    def __init__(self, dataroot, filelist, patch_size, is_train) -> None:
        super(LongMriFullDataset, self).__init__()
        self.dataroot = dataroot
        self.patient_handles = []
        self.scan_list = []
        self.filelist = filelist
        self.max_num_series = 10
        self.maximal_period = 96
        self.input_num = 3
        self.patch_size = patch_size
        ## data augumentation
        self.shift = randomShiftNeo(self.patch_size)
        self.flip = randomFlipEX()
        self.istrain = is_train
        with open(os.path.join(self.dataroot, self.filelist)) as f:
            self.patientlist = f.read().splitlines()
        self.__load_images()
        self.__get_indexes()
        
    def __load_images(self):
        for patient in self.patientlist:
            #scan_list = []
            patient_dir = os.path.join(self.dataroot, patient)
            scans = os.listdir(patient_dir)
            scans = [scan.split('.')[0] for scan in scans if scan.endswith('.nii.gz') and not scan.endswith('_seg.nii.gz') and not scan.endswith('_sdf.nii.gz')]
            scans.sort()
            print(patient, scans)
            MRI = LongMRIDenseSdf(patient_dir, scans)
            self.patient_handles.append(MRI)
    
    def __get_indexes(self):
        for handle in self.patient_handles:
            num_series = handle.get_num_series()
            combs = itertools.combinations(range(num_series),3)
            for comb in combs:
                sample_index = (handle,) + comb
                self.scan_list.append(sample_index)
                #print(sample_index)

    def __getitem__(self, index):
        patient_handle = self.scan_list[index][0]
        num_series = patient_handle.get_num_series()
        assert num_series <= self.max_num_series, "We only accept patient with atmost 10 scans"
        img_list = []
        pnts_list = []
        normals_list = []
        segs_list = []
        sample_coord_list = []
        sample_sdf_list = []
        study_dates = []
        for i in range(self.input_num):
            scan_id = self.scan_list[index][i+1]
            image, seg, densesdf, p, study_date = patient_handle.get_img_data_tp(scan_id)
            image = np.transpose(image, (2, 1 ,0))
            img_list.append(np.expand_dims(image,axis=0))
            
            seg = np.transpose(seg, (2, 1, 0))
            seg[seg > 1] = 1  ## only keep one
            segs_list.append(np.expand_dims(seg, axis=0))
            
            pnts = p[:,0:3]
            pnts_list.append(np.expand_dims(pnts,axis=0))
            normals = p[:,3:6]
            normals_list.append(np.expand_dims(normals,axis=0))

            sample_coord = densesdf[:,0:3]
            sample_sdf = densesdf[:,3]
            sample_coord_list.append(np.expand_dims(sample_coord,axis=0))
            sample_sdf_list.append(np.expand_dims(sample_sdf,axis=0))
            
            study_dates.append(study_date)

        imgs = np.concatenate(img_list)
        imgs = torch.from_numpy(imgs)
        seg = np.concatenate(segs_list,axis=0)
        seg = torch.from_numpy(seg)
        
        s , d, h, w = seg.shape
        _,n_p, _ =  pnts_list[0].shape
        _,n_s, _ =  sample_coord_list[0].shape
        pnts = np.concatenate(pnts_list, axis=0)
        pnts = torch.from_numpy(pnts)
        normals = np.concatenate(normals_list, axis=0)
        normals = torch.from_numpy(normals)
        sample_coord = np.concatenate(sample_coord_list, axis=0)
        sample_coord = torch.from_numpy(sample_coord)
        sample_sdf = np.concatenate(sample_sdf_list, axis=0)
        sample_sdf = torch.from_numpy(sample_sdf)
        if self.istrain:
            data_tensor = torch.concat([imgs,seg],dim=0)
            pnts = rearrange(pnts,'s n c-> (s n) c')
            sample_coord = rearrange(sample_coord, 's n c -> (s n) c')
            pnts, sample_coord, data_tensor = self.flip(pnts, sample_coord, data_tensor)
            pnts, sample_coord, data_tensor = self.shift(pnts, sample_coord, data_tensor)
            imgs = data_tensor[:3,:]
            seg = data_tensor[3:,:]
            
            sample_coord = rearrange(sample_coord,'(s n) c -> s n c',s=self.input_num, n=n_s) 
            pnts = rearrange(pnts,'(s n) c -> s n c',s=self.input_num,n=n_p)
            sample_coord_selected_list = []
            sample_sdf_selected_list = []
            for i in range(self.input_num):
                sample_coord_i = sample_coord[i]
                sample_sdf_i = sample_sdf[i]
                valid_index = torch.all(torch.logical_and(sample_coord_i <= 1,sample_coord_i >= -1), dim=1)
                sample_coord_i = sample_coord_i[valid_index]
                sample_sdf_i = sample_sdf_i[valid_index]
                sample_size = 50000
                random_indices = torch.randperm(sample_coord_i.shape[0])[:sample_size]
            
                sample_coord_tmp = sample_coord_i[random_indices]
                sample_sdf_tmp = sample_sdf_i[random_indices]
                while sample_coord_tmp.shape[0] < sample_size:
                    rest_sample = sample_size - sample_coord_tmp.shape[0]
                    random_indices = torch.randperm(sample_coord_i.shape[0])[:rest_sample]
                    sample_coord_tmp = torch.concat([sample_coord_tmp,sample_coord_i[random_indices]],dim=0)
                    sample_sdf_tmp = torch.concat([sample_sdf_tmp,sample_sdf_i[random_indices]],dim=0)
            
                sample_coord_i = sample_coord_tmp
                sample_sdf_i = sample_sdf_tmp
                assert sample_coord_i.shape[0]>=50000, "too few data"
                
                #sample_coord_i = sample_coord_i[random_indices]
                #sample_sdf_i = sample_sdf_i[random_indices]
                sample_coord_selected_list.append(sample_coord_i.unsqueeze(0))
                sample_sdf_selected_list.append(sample_sdf_i.unsqueeze(0))
            sample_coord = torch.concat(sample_coord_selected_list,dim=0)
            sample_sdf = torch.concat(sample_sdf_selected_list,dim=0)
        
        xyz_coords = np.mgrid[:d, :h, :w].astype(np.float32)
        xyz_coords[0, ...] = xyz_coords[0, ...] / (d - 1)
        xyz_coords[1, ...] = xyz_coords[1, ...] / (h - 1)
        xyz_coords[2, ...] = xyz_coords[2, ...] / (w - 1)
        xyz_coords = (xyz_coords - 0.5) * 2
        xyz_coords = rearrange(xyz_coords, 'c d h w -> 1 (d h w) c')
        xyz_coords = repeat(xyz_coords, '1 d c -> s d c', s=s)
        grid_coord = torch.from_numpy(xyz_coords)
        grid_occ = rearrange(seg,'s d h w -> s (d h w)')
        
        sample_occ = torch.zeros_like(sample_sdf)
        sample_occ[sample_sdf < 0] = 1
        
        imgs = imgs * 2 - 1
        study_dates = np.array(study_dates)
        study_dates[study_dates > self.maximal_period] = self.maximal_period
        study_dates = study_dates / self.maximal_period
        study_dates = torch.from_numpy(study_dates)
        input_dict = {'pnts':pnts,
                      'normals':normals, 
                      'seg': seg,
                      'img':imgs, 
                      'study_dates': study_dates, 
                      'sample_coord':sample_coord,
                      'sample_sdf': sample_sdf,
                      'sample_occ':sample_occ,
                      'grid_coord':grid_coord,
                      'grid_occ':grid_occ
                      }
        return input_dict
    
    def __len__(self):
        return len(self.scan_list)

class LongMriFullDatasetMore(Dataset):
    ##### use first two scans as input and pick one of the latter scan as output
    def __init__(self, dataroot, filelist, patch_size, is_train) -> None:
        super(LongMriFullDatasetMore, self).__init__()
        self.dataroot = dataroot
        self.patient_handles = []
        self.scan_list = []
        self.filelist = filelist
        self.max_num_series = 10
        self.maximal_period = 96
        self.input_num = 3
        self.patch_size = patch_size
        ## data augumentation
        self.shift = randomShiftNeo(self.patch_size)
        self.flip = randomFlipEX()
        self.istrain = is_train
        with open(os.path.join(self.dataroot, self.filelist)) as f:
            self.patientlist = f.read().splitlines()
        self.__load_images()
        self.__get_indexes()
        
    def __load_images(self):
        for patient in self.patientlist:
            #scan_list = []
            patient_dir = os.path.join(self.dataroot, patient)
            scans = os.listdir(patient_dir)
            scans = [scan.split('.')[0] for scan in scans if scan.endswith('.nii.gz') and not scan.endswith('_seg.nii.gz') and not scan.endswith('_sdf.nii.gz')]
            scans.sort()
            print(patient, scans)
            MRI = LongMRIDenseSdf(patient_dir, scans)
            self.patient_handles.append(MRI)
    
    def __get_indexes(self):
        for handle in self.patient_handles:
            num_series = handle.get_num_series()
            for i in range(num_series-2):
                sample_index = (handle,0,1,) + (i+2,)
                self.scan_list.append(sample_index)
                print(sample_index)
    def __getitem__(self, index):
        patient_handle = self.scan_list[index][0]
        num_series = patient_handle.get_num_series()
        assert num_series <= self.max_num_series, "We only accept patient with atmost 10 scans"
        img_list = []
        pnts_list = []
        normals_list = []
        segs_list = []
        sample_coord_list = []
        sample_sdf_list = []
        study_dates = []
        for i in range(self.input_num):
            scan_id = self.scan_list[index][i+1]
            image, seg, densesdf, p, study_date = patient_handle.get_img_data_tp(scan_id)
            image = np.transpose(image, (2, 1 ,0))
            img_list.append(np.expand_dims(image,axis=0))
            
            seg = np.transpose(seg, (2, 1, 0))
            seg[seg > 1] = 1  ## only keep one
            segs_list.append(np.expand_dims(seg, axis=0))
            
            pnts = p[:,0:3]
            pnts_list.append(np.expand_dims(pnts,axis=0))
            normals = p[:,3:6]
            normals_list.append(np.expand_dims(normals,axis=0))

            sample_coord = densesdf[:,0:3]
            sample_sdf = densesdf[:,3]
            sample_coord_list.append(np.expand_dims(sample_coord,axis=0))
            sample_sdf_list.append(np.expand_dims(sample_sdf,axis=0))
            
            study_dates.append(study_date)

        imgs = np.concatenate(img_list)
        imgs = torch.from_numpy(imgs)
        seg = np.concatenate(segs_list,axis=0)
        seg = torch.from_numpy(seg)
        
        s , d, h, w = seg.shape
        _,n_p, _ =  pnts_list[0].shape
        _,n_s, _ =  sample_coord_list[0].shape
        pnts = np.concatenate(pnts_list, axis=0)
        pnts = torch.from_numpy(pnts)
        normals = np.concatenate(normals_list, axis=0)
        normals = torch.from_numpy(normals)
        sample_coord = np.concatenate(sample_coord_list, axis=0)
        sample_coord = torch.from_numpy(sample_coord)
        sample_sdf = np.concatenate(sample_sdf_list, axis=0)
        sample_sdf = torch.from_numpy(sample_sdf)
        if self.istrain:
            data_tensor = torch.concat([imgs,seg],dim=0)
            pnts = rearrange(pnts,'s n c-> (s n) c')
            sample_coord = rearrange(sample_coord, 's n c -> (s n) c')
            pnts, sample_coord, data_tensor = self.flip(pnts, sample_coord, data_tensor)
            pnts, sample_coord, data_tensor = self.shift(pnts, sample_coord, data_tensor)
            imgs = data_tensor[:3,:]
            seg = data_tensor[3:,:]
            
            sample_coord = rearrange(sample_coord,'(s n) c -> s n c',s=self.input_num, n=n_s) 
            pnts = rearrange(pnts,'(s n) c -> s n c',s=self.input_num,n=n_p)
            sample_coord_selected_list = []
            sample_sdf_selected_list = []
            for i in range(self.input_num):
                sample_coord_i = sample_coord[i]
                sample_sdf_i = sample_sdf[i]
                valid_index = torch.all(torch.logical_and(sample_coord_i <= 1,sample_coord_i >= -1), dim=1)
                sample_coord_i = sample_coord_i[valid_index]
                sample_sdf_i = sample_sdf_i[valid_index]
                sample_size = 50000
                random_indices = torch.randperm(sample_coord_i.shape[0])[:sample_size]
            
                sample_coord_tmp = sample_coord_i[random_indices]
                sample_sdf_tmp = sample_sdf_i[random_indices]
                while sample_coord_tmp.shape[0] < sample_size:
                    rest_sample = sample_size - sample_coord_tmp.shape[0]
                    random_indices = torch.randperm(sample_coord_i.shape[0])[:rest_sample]
                    sample_coord_tmp = torch.concat([sample_coord_tmp,sample_coord_i[random_indices]],dim=0)
                    sample_sdf_tmp = torch.concat([sample_sdf_tmp,sample_sdf_i[random_indices]],dim=0)
            
                sample_coord_i = sample_coord_tmp
                sample_sdf_i = sample_sdf_tmp
                assert sample_coord_i.shape[0]>=50000, "too few data"
                
                #sample_coord_i = sample_coord_i[random_indices]
                #sample_sdf_i = sample_sdf_i[random_indices]
                sample_coord_selected_list.append(sample_coord_i.unsqueeze(0))
                sample_sdf_selected_list.append(sample_sdf_i.unsqueeze(0))
            sample_coord = torch.concat(sample_coord_selected_list,dim=0)
            sample_sdf = torch.concat(sample_sdf_selected_list,dim=0)
        
        xyz_coords = np.mgrid[:d, :h, :w].astype(np.float32)
        xyz_coords[0, ...] = xyz_coords[0, ...] / (d - 1)
        xyz_coords[1, ...] = xyz_coords[1, ...] / (h - 1)
        xyz_coords[2, ...] = xyz_coords[2, ...] / (w - 1)
        xyz_coords = (xyz_coords - 0.5) * 2
        xyz_coords = rearrange(xyz_coords, 'c d h w -> 1 (d h w) c')
        xyz_coords = repeat(xyz_coords, '1 d c -> s d c', s=s)
        grid_coord = torch.from_numpy(xyz_coords)
        grid_occ = rearrange(seg,'s d h w -> s (d h w)')
        
        sample_occ = torch.zeros_like(sample_sdf)
        sample_occ[sample_sdf < 0] = 1
        
        imgs = imgs * 2 - 1
        study_dates = np.array(study_dates)
        study_dates[study_dates > self.maximal_period] = self.maximal_period
        study_dates = study_dates / self.maximal_period
        study_dates = torch.from_numpy(study_dates)
        input_dict = {'pnts':pnts,
                      'normals':normals, 
                      'seg': seg,
                      'img':imgs, 
                      'study_dates': study_dates, 
                      'sample_coord':sample_coord,
                      'sample_sdf': sample_sdf,
                      'sample_occ':sample_occ,
                      'grid_coord':grid_coord,
                      'grid_occ':grid_occ
                      }
        return input_dict
    
    def __len__(self):
        return len(self.scan_list)
class LongMriFullDatasetThree(Dataset):
    ##### use first two scans as input and pick one of the latter scan as output
    def __init__(self, dataroot, filelist, patch_size, is_train) -> None:
        super(LongMriFullDatasetThree, self).__init__()
        self.dataroot = dataroot
        self.patient_handles = []
        self.scan_list = []
        self.filelist = filelist
        self.max_num_series = 10
        self.maximal_period = 96
        self.input_num = 3
        self.patch_size = patch_size
        ## data augumentation
        self.shift = randomShiftNeo(self.patch_size)
        self.flip = randomFlipEX()
        self.istrain = is_train
        with open(os.path.join(self.dataroot, self.filelist)) as f:
            self.patientlist = f.read().splitlines()
        self.__load_images()
        self.__get_indexes()
        
    def __load_images(self):
        for patient in self.patientlist:
            #scan_list = []
            patient_dir = os.path.join(self.dataroot, patient)
            scans = os.listdir(patient_dir)
            scans = [scan.split('.')[0] for scan in scans if scan.endswith('.nii.gz') and not scan.endswith('_seg.nii.gz') and not scan.endswith('_sdf.nii.gz')]
            scans.sort()
            print(patient, scans)
            MRI = LongMRIDenseSdf(patient_dir, scans)
            self.patient_handles.append(MRI)
    
    def __get_indexes(self):
        for handle in self.patient_handles:
            num_series = handle.get_num_series()
            sample_index = (handle,0,1,2) 
            self.scan_list.append(sample_index)

    def __getitem__(self, index):
        patient_handle = self.scan_list[index][0]
        num_series = patient_handle.get_num_series()
        assert num_series <= self.max_num_series, "We only accept patient with atmost 10 scans"
        img_list = []
        pnts_list = []
        normals_list = []
        segs_list = []
        sample_coord_list = []
        sample_sdf_list = []
        study_dates = []
        for i in range(self.input_num):
            scan_id = self.scan_list[index][i+1]
            image, seg, densesdf, p, study_date = patient_handle.get_img_data_tp(scan_id)
            image = np.transpose(image, (2, 1 ,0))
            img_list.append(np.expand_dims(image,axis=0))
            
            seg = np.transpose(seg, (2, 1, 0))
            seg[seg > 1] = 1  ## only keep one
            segs_list.append(np.expand_dims(seg, axis=0))
            
            pnts = p[:,0:3]
            pnts_list.append(np.expand_dims(pnts,axis=0))
            normals = p[:,3:6]
            normals_list.append(np.expand_dims(normals,axis=0))

            sample_coord = densesdf[:,0:3]
            sample_sdf = densesdf[:,3]
            sample_coord_list.append(np.expand_dims(sample_coord,axis=0))
            sample_sdf_list.append(np.expand_dims(sample_sdf,axis=0))
            
            study_dates.append(study_date)

        imgs = np.concatenate(img_list)
        imgs = torch.from_numpy(imgs)
        seg = np.concatenate(segs_list,axis=0)
        seg = torch.from_numpy(seg)
        
        s , d, h, w = seg.shape
        _,n_p, _ =  pnts_list[0].shape
        _,n_s, _ =  sample_coord_list[0].shape
        pnts = np.concatenate(pnts_list, axis=0)
        pnts = torch.from_numpy(pnts)
        normals = np.concatenate(normals_list, axis=0)
        normals = torch.from_numpy(normals)
        sample_coord = np.concatenate(sample_coord_list, axis=0)
        sample_coord = torch.from_numpy(sample_coord)
        sample_sdf = np.concatenate(sample_sdf_list, axis=0)
        sample_sdf = torch.from_numpy(sample_sdf)
        if self.istrain:
            data_tensor = torch.concat([imgs,seg],dim=0)
            pnts = rearrange(pnts,'s n c-> (s n) c')
            sample_coord = rearrange(sample_coord, 's n c -> (s n) c')
            pnts, sample_coord, data_tensor = self.flip(pnts, sample_coord, data_tensor)
            pnts, sample_coord, data_tensor = self.shift(pnts, sample_coord, data_tensor)
            imgs = data_tensor[:3,:]
            seg = data_tensor[3:,:]
            
            sample_coord = rearrange(sample_coord,'(s n) c -> s n c',s=self.input_num, n=n_s) 
            pnts = rearrange(pnts,'(s n) c -> s n c',s=self.input_num,n=n_p)
            sample_coord_selected_list = []
            sample_sdf_selected_list = []
            for i in range(self.input_num):
                sample_coord_i = sample_coord[i]
                sample_sdf_i = sample_sdf[i]
                valid_index = torch.all(torch.logical_and(sample_coord_i <= 1,sample_coord_i >= -1), dim=1)
                sample_coord_i = sample_coord_i[valid_index]
                sample_sdf_i = sample_sdf_i[valid_index]
                sample_size = 50000
                random_indices = torch.randperm(sample_coord_i.shape[0])[:sample_size]
            
                sample_coord_tmp = sample_coord_i[random_indices]
                sample_sdf_tmp = sample_sdf_i[random_indices]
                while sample_coord_tmp.shape[0] < sample_size:
                    rest_sample = sample_size - sample_coord_tmp.shape[0]
                    random_indices = torch.randperm(sample_coord_i.shape[0])[:rest_sample]
                    sample_coord_tmp = torch.concat([sample_coord_tmp,sample_coord_i[random_indices]],dim=0)
                    sample_sdf_tmp = torch.concat([sample_sdf_tmp,sample_sdf_i[random_indices]],dim=0)
            
                sample_coord_i = sample_coord_tmp
                sample_sdf_i = sample_sdf_tmp
                assert sample_coord_i.shape[0]>=50000, "too few data"
                
                #sample_coord_i = sample_coord_i[random_indices]
                #sample_sdf_i = sample_sdf_i[random_indices]
                sample_coord_selected_list.append(sample_coord_i.unsqueeze(0))
                sample_sdf_selected_list.append(sample_sdf_i.unsqueeze(0))
            sample_coord = torch.concat(sample_coord_selected_list,dim=0)
            sample_sdf = torch.concat(sample_sdf_selected_list,dim=0)
        
        xyz_coords = np.mgrid[:d, :h, :w].astype(np.float32)
        xyz_coords[0, ...] = xyz_coords[0, ...] / (d - 1)
        xyz_coords[1, ...] = xyz_coords[1, ...] / (h - 1)
        xyz_coords[2, ...] = xyz_coords[2, ...] / (w - 1)
        xyz_coords = (xyz_coords - 0.5) * 2
        xyz_coords = rearrange(xyz_coords, 'c d h w -> 1 (d h w) c')
        xyz_coords = repeat(xyz_coords, '1 d c -> s d c', s=s)
        grid_coord = torch.from_numpy(xyz_coords)
        grid_occ = rearrange(seg,'s d h w -> s (d h w)')
        
        sample_occ = torch.zeros_like(sample_sdf)
        sample_occ[sample_sdf < 0] = 1
        
        imgs = imgs * 2 - 1
        study_dates = np.array(study_dates)
        study_dates[study_dates > self.maximal_period] = self.maximal_period
        study_dates = study_dates / self.maximal_period
        study_dates = torch.from_numpy(study_dates)
        input_dict = {'pnts':pnts,
                      'normals':normals, 
                      'seg': seg,
                      'img':imgs, 
                      'study_dates': study_dates, 
                      'sample_coord':sample_coord,
                      'sample_sdf': sample_sdf,
                      'sample_occ':sample_occ,
                      'grid_coord':grid_coord,
                      'grid_occ':grid_occ
                      }
        return input_dict
    
    def __len__(self):
        return len(self.scan_list)

class LongMriPointCloudEXDataset(Dataset):
    ##### use first two scans as input and pick one of the latter scan as output
    def __init__(self, dataroot, filelist, patch_size, is_train) -> None:
        super(LongMriPointCloudEXDataset, self).__init__()
        self.dataroot = dataroot
        self.patient_handles = []
        self.scan_list = []
        self.filelist = filelist
        self.max_num_series = 10
        self.maximal_period = 96
        self.patch_size = patch_size
        ## data augumentation
        self.shift = randomShiftNeo(self.patch_size)
        self.flip = randomFlipEX()
        self.istrain = is_train
        with open(os.path.join(self.dataroot, self.filelist)) as f:
            self.patientlist = f.read().splitlines()
        self.__load_images()
        self.__get_indexes()
        
    def __load_images(self):
        for patient in self.patientlist:
            #scan_list = []
            patient_dir = os.path.join(self.dataroot, patient)
            scans = os.listdir(patient_dir)
            scans = [scan.split('.')[0] for scan in scans if scan.endswith('.nii.gz') and not scan.endswith('_seg.nii.gz') and not scan.endswith('_sdf.nii.gz')]
            scans.sort()
            print(patient, scans)
            MRI = LongMRIDenseSdf(patient_dir, scans)
            self.patient_handles.append(MRI)
    
    def __get_indexes(self):
        for handle in self.patient_handles:
            num_series = handle.get_num_series()
            for i in range(num_series):
                sample_index = (handle, i) 
                self.scan_list.append(sample_index)

    def __getitem__(self, index):
        patient_handle, scan_id = self.scan_list[index]
        num_series = patient_handle.get_num_series()
        assert num_series <= self.max_num_series, "We only accept patient with atmost 10 scans"
        
        image, seg, densesdf, p, study_date = patient_handle.get_img_data_tp(scan_id)
        image = np.transpose(image, (2, 1, 0))
        image = torch.from_numpy(image)
        seg = np.transpose(seg, (2, 1, 0))
        seg = torch.from_numpy(seg)
        seg[seg>=1] = 1
        d, h, w = seg.shape
        p = torch.from_numpy(p)
        pnts = p[:,0:3]
        normals = p[:,3:6]
        sample_coord = densesdf[:,0:3]
        sample_sdf = densesdf[:,3]
        if self.istrain:
            sample_coord = torch.from_numpy(sample_coord)
            sample_sdf = torch.from_numpy(sample_sdf)
            tensor = torch.concat([image.unsqueeze(0), seg.unsqueeze(0)], dim=0)
            pnts, sample_coord, tensor = self.flip(pnts, sample_coord, tensor)
            pnts, sample_coord, tensor = self.shift(pnts, sample_coord, tensor)
            
            valid_index = torch.all(torch.logical_and(sample_coord <= 1,sample_coord >= -1), dim=1)
            sample_coord = sample_coord[valid_index]
            sample_sdf = sample_sdf[valid_index]
            sample_size = 50000
            random_indices = torch.randperm(sample_coord.shape[0])[:sample_size]
            sample_coord_tmp = sample_coord[random_indices]
            sample_sdf_tmp = sample_sdf[random_indices]
            while sample_coord_tmp.shape[0] < sample_size:
                rest_sample = sample_size - sample_coord_tmp.shape[0]
                random_indices = torch.randperm(sample_coord.shape[0])[:rest_sample]
                sample_coord_tmp = torch.concat([sample_coord_tmp,sample_coord[random_indices]],dim=0)
                sample_sdf_tmp = torch.concat([sample_sdf_tmp,sample_sdf[random_indices]],dim=0)
            
            sample_coord = sample_coord_tmp
            sample_sdf = sample_sdf_tmp
            assert sample_coord.shape[0]>=50000, "too few data"
            image = tensor[0,:].unsqueeze(0)
            seg = tensor[1,:].unsqueeze(0)
        else:
            ##### test all the voxels #######
            image = image.unsqueeze(0)
            seg = seg.unsqueeze(0)
            sample_coord = torch.from_numpy(sample_coord)
            sample_sdf = torch.from_numpy(sample_sdf)
        
        xyz_coords = np.mgrid[:d, :h, :w].astype(np.float32)
        xyz_coords[0, ...] = xyz_coords[0, ...] / (d - 1)
        xyz_coords[1, ...] = xyz_coords[1, ...] / (h - 1)
        xyz_coords[2, ...] = xyz_coords[2, ...] / (w - 1)
        xyz_coords = (xyz_coords - 0.5) * 2
        xyz_coords = rearrange(xyz_coords, 'c d h w -> (d h w) c')
        grid_coord = torch.from_numpy(xyz_coords)
        grid_occ = rearrange(seg,'1 d h w -> (d h w)')
        image = image * 2 - 1
        sample_occ = torch.zeros_like(sample_sdf)
        sample_occ[sample_sdf < 0] = 1
        input_dict = {'pnts':pnts,
                      'normals': normals, 
                      'img': image,
                      'seg': seg,
                      'grid_coord':grid_coord,
                      'grid_occ':grid_occ,
                      'sample_coord':sample_coord,
                      'sample_sdf': sample_sdf,
                      'sample_occ':sample_occ
                      }
        return input_dict
    
    def __len__(self):
        return len(self.scan_list)

if __name__=='__main__':
    ROOT_PATH = '/exports/lkeb-hpc/ychen/01_data/08_VS_followup/toy_data/2023-10-18_17h-49m-18s_image_all_tif.h5'
