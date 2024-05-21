from random import sample
import torch
from models.networks.convlstm import Conv3DLSTM, STConv3DLSTM, STConv3DLSTMSmall
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from einops import rearrange
import os
import numpy as np
import util.util as util
from util.util import dice_loss, NormalSampler, dice_coeff3D
import torch.nn.functional as F
from torch.autograd import grad
from datetime import datetime

class NormLoss(torch.nn.Module):
    def __init__(self, p=2):
        """
        p: the order of the norm (e.g., 2 for L2 norm)
        """
        super(NormLoss, self).__init__()
        self.p = p

    def forward(self, input_tensor):
        #batch_size,_,d,w,h = input_tensor.shape
        loss = torch.mean(torch.norm(input_tensor, dim=1, p=self.p))
        return loss

class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass

def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        #only_inputs=True)[0][:, -3:]
        only_inputs=True)[0]
    return points_grad

class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return self.initial * (self.factor ** (epoch // self.interval))

class MultiMriLstmModel(torch.nn.Module):
    def __init__(self, opt, device):
        super().__init__()
        self.opt = opt
        self.device = device
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.do_grap_clipping = False
        #self.sampler = NormalSampler(global_sigma=1.8,local_sigma=0.01)
        
        ### initial the networks
        self.generator = Conv3DLSTM(input_dim=2,hidden_dim=[64,128,64],kernel_size=[(3,3,3),(3,3,3),(3,3,3)],num_layers=3,\
                batch_first=True,bias=True,return_all_layers=False)
        #print('STConv3DLSTM')
        #self.generator = STConv3DLSTM(input_dim=2,hidden_dim=[64,64,64],kernel_size=[(3,3,3),(3,3,3),(3,3,3)],num_layers=3,\
        #        batch_first=True,bias=True,return_all_layers=False)
        self.generator.cuda()
        self.generator.print_network()
        print(self.generator)
        
        self.netD = None
        
        if not opt.isTrain or opt.continue_train:
            self.generator = util.load_network(self.generator, 'Generator', opt.which_epoch, opt)
            if opt.isTrain and opt.use_gan:
                self.netD = util.load_network(self.netD, 'D', opt.which_epoch, opt)
        
        # set loss functions
        if opt.isTrain:
            if opt.latent_code_regularization:
                self.ll_loss = NormLoss(p=2)
            self.criterionRec = torch.nn.MSELoss()
            #self.criterionRec = torch.nn.L1Loss()
        self.criterionSeg = torch.nn.BCEWithLogitsLoss()
        ## set optimizer
        self.scaler = GradScaler()
        beta1, beta2 = opt.beta1, opt.beta2
        E_lr,G_lr = opt.E_lr, opt.lr
        
        G_params = list(self.generator.parameters())
        self.optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2), weight_decay=3e-5)
        
        #self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer_G,\
        #            lr_lambda=LambdaLinear(self.opt.niter + self.opt.niter_decay, self.opt.niter).step)
        self.lr_scheduler_G = StepLearningRateSchedule(G_lr, 50, 0.5)
    
    def compute_generator_loss(self, data, use_gan):
        G_losses = {}
        images = data['img'].float().cuda()
        segs = data['seg'].float().cuda()
        study_dates = data['study_dates'].float().cuda()
        
        #### input and output
        input_length = study_dates.shape[1] - 1
        time_interval = study_dates[:,1:] - study_dates[:,0:2]

        #input_dates = study_dates[:,0:input_length]
        images = images.unsqueeze(2)
        segs = segs.unsqueeze(2)
        data_tensor = torch.concat([images, segs], dim=2) ##### B,T,C,D,H,W
        input_tensor = data_tensor[:,:input_length]
        gt_tensor = data_tensor[:,1:,0]
        gt_mask = data_tensor[:,1:,1]

        pred_img, pred_seg, state = self.generator(input_tensor, time_interval)
        #print(pred_seg.shape, gt_mask.shape)
        #pred_seg = rearrange(pred_seg, 'b s c d h w -> (b s) c d h w')
        gt_tensor = rearrange(gt_tensor,'b s d h w -> b s 1 d h w')
        gt_mask = rearrange(gt_mask,'b s d h w -> b s 1 d h w')
        G_losses['rec_loss'] = self.criterionRec(pred_img.unsqueeze(2), gt_tensor)
        #G_losses['rec_loss'] = self.criterionRec(pred_img, gt_tensor)
        G_losses['seg_loss'] = self.criterionSeg(pred_seg, gt_mask) + \
                dice_loss(F.sigmoid(pred_seg), gt_mask.float(),multiclass=False)
        return G_losses, pred_img
    
    def run_generator_one_step(self, data, use_gan):
        self.optimizer_G.zero_grad()
        g_losses, generated = self.compute_generator_loss(data, use_gan)
        print(g_losses)
        #g_loss = g_losses['rec_loss']  + g_losses['seg_loss']
        g_loss = g_losses['rec_loss'] + 0.1 * g_losses['seg_loss']
        self.scaler.scale(g_loss).backward()
        self.scaler.step(self.optimizer_G)
        self.scaler.update()
        return g_losses, generated

    def run_evaluation_during_training(self, dataloader, grid_size):
        num_val = len(dataloader)
        self.eval()
        batch = None
        generated_img = None
        generated_seg = None
        val_rec_loss = 0
        val_seg_loss = 0
        val_score = 0
        d, h ,w = grid_size
        with torch.no_grad():
            for _, batch in enumerate(dataloader):
                images = batch['img'].float().cuda()
                segs = batch['seg'].float().cuda()
                study_dates = batch['study_dates'].float().cuda()
        
                #### input and output
                input_length = study_dates.shape[1] - 1
                #input_dates = study_dates[:,0:input_length]
                images = images.unsqueeze(2)
                segs = segs.unsqueeze(2)
                data_tensor = torch.concat([images, segs], dim=2) ##### B,T,C,D,H,W
                input_tensor = data_tensor[:,:input_length]
                
                gt_tensor = data_tensor[:,1:,0]
                gt_mask = data_tensor[:,1:,1]
                gt_mask = rearrange(gt_mask,'b s d h w -> b s 1 d h w')

                pred_img, pred_seg, state = self.generator(input_tensor, study_dates)
                #loss = self.criterionSeg(seg, gt_mask)
                loss = self.criterionRec(pred_img, gt_tensor)
                seg_loss = self.criterionSeg(pred_seg, gt_mask) + \
                    dice_loss(F.sigmoid(pred_seg), gt_mask.float(),multiclass=False)
                val_rec_loss += loss
                val_seg_loss += seg_loss
                val_score += dice_coeff3D((torch.sigmoid(pred_seg[:,1,:]) >0.5).float(), gt_mask[:,1,:])

            generated_img = pred_img[0,1]
            generated_seg = pred_seg[0,1]
            generated_seg = torch.sigmoid(generated_seg)>0.5
            generated_seg = generated_seg.float()
            seg_gt = batch['seg'][0][2].float()
            img_gt = batch['img'][0][2].float()
        self.train()
        #generated_img = rearrange(generated_img, '1 1 (d h w) ->d h w',d=d,h=h,w=w)
        return img_gt, seg_gt, generated_img, generated_seg, val_rec_loss / num_val, val_seg_loss / num_val, val_score / num_val

    def update_learning_rate(self, epoch):
        for _, param_group in enumerate(self.optimizer_G.param_groups):
            param_group["lr"] = self.lr_scheduler_G.get_learning_rate(epoch)
    
    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
    
    def save(self, epoch):
        util.save_network(self.generator, 'Generator', epoch, self.opt)
        
        state = {'epochs': epoch, 'n_epochs': self.opt.niter + self.opt.niter_decay}
        state['Generator_opt']= self.optimizer_G.state_dict()

        torch.save(state, os.path.join(self.opt.checkpoints_dir, self.opt.name, 'checkpoint.pth.tar'))
