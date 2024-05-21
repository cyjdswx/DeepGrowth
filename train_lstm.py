import sys,os
from data.mrisdf_dataset import LongMriImageDataset, LongMriImageCompleteDataset, LongMriImageMoreDataset, LongMriImageClipDataset
from models.Multimrilstm_model import MultiMriLstmModel
import torch
import time
from torch.backends import cudnn
from torch.utils.data import DataLoader
import random
import numpy as np
import argparse
import wandb

from models.networks import F

def my_collate(batch):
    ##### collate function for variable-size seires
    input_img = [item['img'] for item in batch]
    input_sdf = [item['seg'] for item in batch]
    samples_coord = [item['samples_coord'] for item in batch]
    samples_sdf = [item['samples_sdf'] for item in batch]
    return [input_img, input_sdf, samples_coord, samples_sdf]

def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.use_deterministic_algorithms(True)
    cudnn.deterministic = True
    cudnn.benchmark = False

if __name__=='__main__':
    # parse options
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # experiment specifics
    parser.add_argument('--name', type=str, default='brats_t1ce', help='name of the experiment')
    parser.add_argument('--dataset_dir', type=str, default='/exports/lkeb-hpc/ychen/01_data/08_VS_followup/longitudinal_t1ce_whole_cropped96')
    parser.add_argument('--nThreads', default=12, type=int, help='# threads for loading data')
    parser.add_argument('--load_from_opt_file', action='store_true', help='load the options from checkpoints and use that as default')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    
    parser.add_argument('--norm_G', type=str, default='instanceaffine', help='instance normalization or batch normalization')
    parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

    # input/output sizes
    parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
    parser.add_argument('--patch_height', type=int, default=64, help='# of height of image patch')
    parser.add_argument('--patch_width', type=int, default=64, help='# of width of image patch')
    parser.add_argument('--patch_thickness', type=int, default=64, help='# of slices of image patch')
    parser.add_argument('--maximal_period', type=float, default=96, help='# of months')
    
    # Hyperparameters
    parser.add_argument('--lr_width', type=int, default=64, help='low res stream strided conv number of channles')
    parser.add_argument('--lr_max_width', type=int, default=1024, help='low res stream conv number of channles')
    parser.add_argument('--lr_depth', type=int, default=7, help='low res stream number of conv layers')
    parser.add_argument('--hr_width', type=int, default=64, help='high res stream number of MLP channles')
    parser.add_argument('--hr_depth', type=int, default=5, help='high res stream number of MLP layers')
    parser.add_argument('--latent_dim', type=int, default=64, help='high res stream number of MLP layers')
    parser.add_argument('--reflection_pad', action='store_true', help='if specified, use reflection padding at lr stream')
    parser.add_argument('--replicate_pad', action='store_true', help='if specified, use replicate padding at lr stream')
    parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')
    parser.add_argument('--hr_coor', choices=('cosine', 'None','siren'), default='cosine')

    parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')
    parser.add_argument('--use_gan', action='store_true', help='enable training with an image encoder.')

    # for training
    parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
    parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--niter', type=int, default=150, help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
    parser.add_argument('--niter_decay', type=int, default=201, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
    parser.add_argument('--no_TTUR', action='store_true', help='Use TTUR training scheme')
    parser.add_argument('--fold', type=int,default=0, help='Use TTUR training scheme')

    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--E_lr', type=float, default=0.0002, help='initial learning rate for adam')

    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
    parser.add_argument('--lambda_rec', type=float, default=1000.0, help='weight for L1 loss')
    parser.add_argument('--lambda_ll', type=float, default=10.0, help='weight for L1 loss')
    parser.add_argument('--no_adv_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
    parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
    parser.add_argument('--latent_code_regularization', action='store_true', help='if specified, use weight decay loss on the estimated parameters from LR')
    parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
    
    # for discriminators
    parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image)')
    parser.add_argument('--lambda_kld', type=float, default=0.05)
    parser.add_argument('--netD_subarch', type=str, default='n_layer', help='architecture of each discriminator')
    parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to be used in multiscale')
    parser.add_argument('--n_layers_D', type=int, default=4, help='# layers in each discriminator')
    parser.add_argument('--ndf_max', type=int, default=512, help='maximal number of discriminator filters')

    # print options to help debugging
    opt = parser.parse_args()
    opt.isTrain = True   # train or test
    setup_seed(100)
    
    #################### set gpu ids  ####################
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])

    assert len(opt.gpu_ids) == 0 or opt.batchSize % len(opt.gpu_ids) == 0, \
            "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
            % (opt.batchSize, len(opt.gpu_ids))
    #################### print configs ###################
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
    print(' '.join(sys.argv))
    
    wandb.login()
    wandb.init(
            project="LongMRItumorneo",
            name=opt.name,
            config=vars(opt)#,
            #mode='disabled'
            )
    
    patch_height = opt.patch_height
    patch_width = opt.patch_width
    patch_depth = opt.patch_thickness
    data_root = opt.dataset_dir
    fold = opt.fold
    device = torch.device('cpu' if opt.gpu_ids == -1 else 'cuda')
    train_dataroot = os.path.join(data_root, 'train_data')
    

    listfile = 'patientlist_fold' + str(fold) + '_train.txt'
    train_instance = LongMriImageDataset(data_root, listfile, (patch_depth, patch_height, patch_width), is_train=True)
    #train_instance = LongMriImageClipDataset(data_root, listfile, (patch_depth, patch_height, patch_width), is_train=True)

    print("dataset [%s] of size %d was created" %
            (type(train_instance).__name__, len(train_instance)))
    dataloader = DataLoader(
        train_instance,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.nThreads),
        drop_last=opt.isTrain
    )
    listfile = 'patientlist_fold' + str(fold) + '_val.txt'
    val_instance = LongMriImageDataset(data_root,listfile, (patch_depth, patch_height, patch_width), is_train=False)
    #val_instance = LongMriImageClipDataset(data_root,listfile, (patch_depth, patch_height, patch_width), is_train=False)
    print("dataset [%s] of size %d was created" %
            (type(val_instance).__name__, len(val_instance)))

    val_dataloader = DataLoader(
        val_instance,
        #batch_size=opt.batchSize,
        batch_size=1,
        shuffle=True,
        drop_last=opt.isTrain
    )
    #### create checkpoints directory ####
    if not os.path.isdir(opt.checkpoints_dir):
        os.mkdir(opt.checkpoints_dir)
    experiment_dir = os.path.join(opt.checkpoints_dir, opt.name)
    if not os.path.isdir(experiment_dir):
        os.mkdir(experiment_dir)
    
    # initialize trainer
    model = MultiMriLstmModel(opt, device) 
    batches_per_epoch = 50

    if opt.use_gan:
        use_gan = True
    else:
        use_gan = False
    
    dataiter = iter(dataloader)
    total_epochs = opt.niter + opt.niter_decay
    start_epoch = 1
    
    for epoch in range(1, total_epochs + 1):
        epoch_start_time = time.time()
        print("epoch:%d" % epoch)
        rec_loss = 0
        seg_loss = 0
        mnf_loss = 0
        grad_loss = 0 
        normal_loss = 0
        latent_loss = 0
        for i in range(batches_per_epoch):
            try:
                data_i = next(dataiter)
            except StopIteration:
                dataiter = iter(dataloader)
                data_i = next(dataiter)
            # train the generator
            g_losses, pred_img = model.run_generator_one_step(data_i, use_gan)
            rec_loss += g_losses['rec_loss']
            seg_loss += g_losses['seg_loss']
        model.update_learning_rate(epoch)
        logs = {}
        logs['rec_loss'] = rec_loss / batches_per_epoch
        logs['seg_loss'] = seg_loss / batches_per_epoch
        #if opt.latent_code_regularization:
        #    logs['latent_loss'] = latent_loss / batches_per_epoch
        for i, param_group in enumerate(model.optimizer_G.param_groups):
            logs['lr_' + str(i)] = param_group['lr']
        
        #### validation
        #if False:
        if val_dataloader is not None:
            img_gt, seg_gt, val_output, val_seg, val_rec_loss, val_seg_loss,val_score = model.run_evaluation_during_training(val_dataloader, (64,64,64))
            logs['val_rec_loss'] = val_rec_loss
            logs['val_seg_loss'] = val_seg_loss
            logs['val_score'] = val_score

            pred_seg = val_seg.squeeze().unsqueeze(1)
            logs['pred_seg'] = wandb.Image(pred_seg)
            pred_img = val_output.squeeze().unsqueeze(1)
            logs['pred_img'] = wandb.Image(pred_img)
            seg_image = seg_gt.squeeze().unsqueeze(1)
            logs['real_seg'] = wandb.Image(seg_image)
            real_image =img_gt.squeeze().unsqueeze(1)
            logs['real_image'] = wandb.Image(real_image)
        wandb.log(logs)
        
        time_per_epoch = time.time() - epoch_start_time
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, total_epochs, time_per_epoch))
        model.save(epoch)
        print('saving the latest model (epoch %d)' % epoch)
        model.save('latest')
    print('Training was successfully finished.')
    
    wandb.finish()
