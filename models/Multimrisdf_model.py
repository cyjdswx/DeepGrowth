from random import sample
import time
from scipy.stats import studentized_range
import torch
from models.networks.encoder import TensorDownEncoderLSTMNot, TensorDownEncoderLSTMET, TensorEncoderLSTM,  TensorDownX4EncoderLSTMNot, TensorDownX4EncoderLSTM, TensorDownX4EncoderLSTMold
from models.networks.decoder import NFCoordConcatDecoder3DMulti,NFCoordConcatDecoder3D, NFCoordSirenMultiDecoder3D
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from util.util import dice_loss, NormalSampler, dice_coeff3D
from einops import rearrange
import os
import numpy as np
import util.util as util
from torch.autograd import grad
import nibabel as nib

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

class MultimrisdfModel(torch.nn.Module):
    def __init__(self, opt, device):
        super().__init__()
        self.opt = opt
        self.device = device
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.do_grap_clipping = False
        
        ### initial the networks
        #self.Encoder = TensorDownX4EncoderLSTMNot(num_in=3,dim=32, hidden_dim=32,n_blocks=5)
        #self.Encoder = TensorDownX4EncoderLSTML4Not(num_in=3,dim=32, hidden_dim=32,n_blocks=5)
        self.Encoder = TensorDownX4EncoderLSTM(num_in=3,dim=32, hidden_dim=32,n_blocks=5,order=5)
        #self.Encoder = TensorDownX8EncoderLSTM(num_in=3,dim=32, hidden_dim=32,n_blocks=5,order=5)
        #self.Encoder = TensorDownEncoderLSTMNot(num_in=3,dim=32, hidden_dim=32,n_blocks=5)
        #self.Encoder = TensorDownEncoderLSTMET(num_in=3,dim=32, hidden_dim=32,n_blocks=5, order=5)
        #self.Encoder = TensorEncoderLSTM(num_in=3,dim=32, hidden_dim=32,n_blocks=5, order=5)
        self.Encoder.cuda()
        self.Encoder.print_network()
        if opt.pretrained is not None:
            self.Encoder=util.load_pretrained_network(self.Encoder,'Encoder','latest',opt, False)
        print(self.Encoder)
        '''
        ## freeze encoder
        for name, params in self.Encoder.named_parameters():
            if 'convlstm' not in name:
                print(name)
                params.requires_grad = False
        '''
        self.dims = [64,64,64,64]
        self.Decoder = NFCoordSirenMultiDecoder3D(num_outputs=1, latent_dim=32,dims=self.dims,maximal_period=96,last_tanh=True)
        self.Decoder.cuda()
        self.Decoder.print_network()
        if opt.pretrained is not None:
            self.Decoder=util.load_pretrained_network(self.Decoder,'Decoder','latest',opt, True)
        print(self.Decoder)
        '''
        ## freeze decoder
        for param in self.Decoder.parameters():
            param.requires_grad = False
        '''
        self.netD = None
        

        if not opt.isTrain or opt.continue_train:
            self.Encoder = util.load_network(self.Encoder, 'Encoder', opt.which_epoch, opt)
            self.Decoder = util.load_network(self.Decoder, 'Decoder', opt.which_epoch, opt)
            if opt.isTrain and opt.use_gan:
                self.netD = util.load_network(self.netD, 'D', opt.which_epoch, opt)
        
        # set loss functions
        if opt.isTrain:
            if opt.latent_code_regularization:
                self.ll_loss = NormLoss(p=2)
            #self.criterionRec = torch.nn.MSELoss()
            self.criterionRec = torch.nn.L1Loss()
            self.criterionSeg = torch.nn.L1Loss()

        ## set optimizer
        self.scaler = GradScaler()
        beta1, beta2 = opt.beta1, opt.beta2
        E_lr,G_lr = opt.E_lr, opt.lr
        E_params = list(self.Encoder.parameters())
        self.optimizer_E = torch.optim.Adam(E_params, lr=E_lr, betas=(beta1, beta2), weight_decay=3e-5)
        G_params = list(self.Decoder.parameters())
        self.optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2), weight_decay=3e-5)
        
        self.lr_scheduler_E = StepLearningRateSchedule(E_lr, 50, 0.5)
        self.lr_scheduler_G = StepLearningRateSchedule(G_lr, 50, 0.5)
    
    def compute_generator_loss(self, data, use_gan):
        G_losses = {}
        pcloud = data['pnts'].float().cuda()
        imgs = data['img'].float().cuda()
        segs = data['seg'].float().cuda()
        #normals = data['normals'].float().cuda()
        sample_coord = data['sample_coord'].float().cuda()
        sample_sdf = data['sample_sdf'].float().cuda()
        study_dates = data['study_dates'].float().cuda()
        bs,s,d,h,w = segs.shape
        #### input and output
        input_tensor = torch.concat([imgs.unsqueeze(2), segs.unsqueeze(2)],dim=2)
        time_interval = study_dates[:,1:] - study_dates[:,0:2]
        features_ori, features_lstm = self.Encoder(input_tensor, time_interval)
        
        generated_sdf_ori = self.Decoder(sample_coord, features_ori)
        generated_sdf_pred = self.Decoder(sample_coord[:,1:,:], features_lstm)
        
        G_losses['seg_loss_1'] = self.criterionSeg(generated_sdf_ori, sample_sdf)
        G_losses['seg_loss_2'] = self.criterionSeg(generated_sdf_pred, sample_sdf[:,1:,:])
        
        if self.opt.latent_code_regularization:
            G_losses['latent_loss'] = self.ll_loss(features_lstm) + self.ll_loss(features_ori)
        return G_losses, generated_sdf_pred
    
    def run_generator_one_step(self, data, use_gan):
        self.optimizer_G.zero_grad()
        self.optimizer_E.zero_grad()
        g_losses, generated = self.compute_generator_loss(data, use_gan)
        print(g_losses)
        g_loss = g_losses['seg_loss_1'] + g_losses['seg_loss_2'] + self.opt.lambda_ll * g_losses['latent_loss']# +g_losses['fm_loss']
        self.scaler.scale(g_loss).backward()
        self.scaler.step(self.optimizer_E)
        self.scaler.step(self.optimizer_G)
        self.scaler.update()
        return g_losses, generated

    def run_evaluation_during_trainingATT(self, dataloader, grid_size):
        num_val = len(dataloader)
        #print('num_val',num_val)
        self.eval()
        batch = None
        generated_sdf = None
        val_rec_loss_1 = 0
        val_rec_loss_2 = 0
        val_fm_loss = 0
        val_score = 0
        d, h ,w = grid_size
        
        with torch.no_grad():
            for _, batch in enumerate(dataloader):
                pcloud = batch['pnts'].float().cuda()
                imgs = batch['img'].float().cuda()
                segs = batch['seg'].float().cuda()
                sample_coord = batch['sample_coord'].float().cuda()
                sample_sdf = batch['sample_sdf'].float().cuda()
                #sample_occ = batch['sample_occ'].float().cuda()
                study_dates = batch['study_dates'].float().cuda()
                
                grid_coord = batch['grid_coord'].float().cuda()
                grid_occ = batch['grid_occ'].float().cuda()
                
                bs,s,d,h,w = segs.shape
                #### input and output
                input_tensor = torch.concat([imgs.unsqueeze(2), segs.unsqueeze(2)],dim=2)
                #time_interval = study_dates[:,1:] - study_dates[:,:-1]
                time_interval = study_dates[:,1:] - study_dates[:,0:2]

                features_ori,features_pred = self.Encoder(input_tensor,time_interval)
                #features = torch.concat([features_mix[:,0:2,:],features_lstm[:,1,:].unsqueeze(1)],dim=1)
                
                #generated_sdf = self.Decoder(sample_coord, features)
                generated_sdf_ori = self.Decoder(sample_coord, features_ori)
                #generated_sdf_pred = self.Decoder(sample_coord[:,2,:].unsqueeze(1), features_pred[:,1,:].unsqueeze(1))
                generated_sdf_pred = self.Decoder(sample_coord[:,1:,:], features_pred)
                loss_1 = self.criterionSeg(generated_sdf_ori, sample_sdf)
                #loss_2 = self.criterionSeg(generated_sdf_pred, sample_sdf[:,2,:])
                loss_2 = self.criterionSeg(generated_sdf_pred, sample_sdf[:,1:,:])
                
                generated_occ = self.Decoder(grid_coord[:,2,:].unsqueeze(1), features_pred[:,1,:].unsqueeze(1))
                generated_occ = rearrange(generated_occ,'bs 1 (d h w)-> bs 1 d h w',d=d,h=h,w=w)
                grid_occ_2 = rearrange(grid_occ[:,2,:], 'bs (d h w) -> bs 1 d h w',d=d,h=h,w=w)
                val_score += dice_coeff3D((generated_occ < 0).float(), grid_occ_2)
                loss_fm = 0
                #loss_fm = self.criterionRec(features_ori[:,1:,:], features_pred)
                #loss_fm = self.criterionRec(features_ori[:,2,:], features_pred[:,1,:])
                
                val_rec_loss_1 += loss_1
                val_rec_loss_2 += loss_2
                val_fm_loss += loss_fm
            generated_sdf = generated_occ[0][0]
            generated_seg = generated_sdf < 0
            generated_seg = generated_seg.float()
            img_gt = batch['img'][0][-1].float()
            seg_gt = batch['seg'][0][-1].float()
        self.train()
        #generated_sdf = rearrange(generated_sdf, '1 1 d h w->d h w',d=d,h=h,w=w)
        #generated_seg = rearrange(generated_seg, '1 1 d h w ->d h w',d=d,h=h,w=w)
        return img_gt, seg_gt, generated_sdf, generated_seg, val_rec_loss_1 / num_val, val_rec_loss_2 / num_val, val_fm_loss / num_val, val_score / num_val

    def update_learning_rate(self, epoch):
        for _, param_group in enumerate(self.optimizer_G.param_groups):
            param_group["lr"] = self.lr_scheduler_G.get_learning_rate(epoch)
        for _, param_group in enumerate(self.optimizer_E.param_groups):
            param_group["lr"] = self.lr_scheduler_E.get_learning_rate(epoch)
    
    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
    
    def save(self, epoch):
        util.save_network(self.Decoder, 'Decoder', epoch, self.opt)
        util.save_network(self.Encoder, 'Encoder', epoch, self.opt)
        
        state = {'epochs': epoch, 'n_epochs': self.opt.niter + self.opt.niter_decay}
        state['Decoder_opt']= self.optimizer_G.state_dict()
        state['Encoder_opt']= self.optimizer_E.state_dict()

        torch.save(state, os.path.join(self.opt.checkpoints_dir, self.opt.name, 'checkpoint.pth.tar'))
