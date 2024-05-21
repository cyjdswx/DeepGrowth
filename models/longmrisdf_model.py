import torch
from models.networks.encoder import TensorEncoder, TensorDownEncoder,TensorDownX4Encoder, TensorDownX8Encoder
from models.networks.decoder import NFCoordConcatDecoder3D, NFCoordSirenSimpleDecoder3D, NFCoordSirenDecoder3D, NFCoordSirenModulationDecoder3D
from torch.cuda.amp import GradScaler
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from einops import rearrange
from util.util import dice_loss, NormalSampler, dice_coeff3D
import os
import numpy as np
import util.util as util
from util.util import LambdaLinear, NormalSampler
from torch.autograd import grad

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

class LongMrisdfModel(torch.nn.Module):
    def __init__(self, opt, device):
        super().__init__()
        self.opt = opt
        self.device = device
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.do_grap_clipping = False
        ### initial the networks
        #self.Encoder = TensorEncoder(num_in=3, dim=32, hidden_dim=32,n_blocks=5)
        #self.Encoder = ViTEncoder(image_size=64,image_patch_size=8,frames=64,frame_patch_size=8,\
        #                num_classes=64,dim=512,depth=6,heads=8,mlp_dim=1024,dim_head=64)
        #self.Encoder = TensorDownEncoder(num_in=3, dim=32, hidden_dim=32,n_blocks=5)
        self.Encoder = TensorDownX4Encoder(num_in=2, dim=32, hidden_dim=32,n_blocks=5)
        #self.Encoder = TensorDownX8Encoder(num_in=2, dim=32, hidden_dim=32,n_blocks=5)
        
        self.Encoder.cuda()
        self.Encoder.print_network()
        print(self.Encoder)
        
        self.Decoder = NFCoordSirenDecoder3D(num_outputs=1,latent_dim=32,last_tanh=True)
        self.Decoder.cuda()
        self.Decoder.print_network()
        print(self.Decoder)

        self.netD = None
        
        if not opt.isTrain or opt.continue_train:
            self.Decoder = util.load_network(self.Decoder, 'G', opt.which_epoch, opt)
            if opt.isTrain and opt.use_gan:
                self.netD = util.load_network(self.netD, 'D', opt.which_epoch, opt)
        
        # set loss functions
        if opt.isTrain:
            if opt.latent_code_regularization:
                self.ll_loss = NormLoss(p=2)
            self.criterionRec = torch.nn.L1Loss()
            self.criterionSeg = torch.nn.L1Loss()
            #self.criterionRec = torch.nn.L1Loss(reduction='sum')

        ## set optimizer
        self.scaler = GradScaler()
        beta1, beta2 = opt.beta1, opt.beta2
        E_lr,G_lr = opt.E_lr, opt.lr
        E_params = list(self.Encoder.parameters())
        self.optimizer_E = torch.optim.Adam(E_params, lr=E_lr, betas=(beta1, beta2), weight_decay=3e-5)
        G_params = list(self.Decoder.parameters())
        self.optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2), weight_decay=3e-5)
        
        #self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer_G,\
        #            lr_lambda=LambdaLinear(self.opt.niter + self.opt.niter_decay, self.opt.niter).step)
        self.lr_scheduler_E = StepLearningRateSchedule(E_lr, 50, 0.5)
        self.lr_scheduler_G = StepLearningRateSchedule(G_lr, 50, 0.5)
    
    def compute_generator_loss(self, data, use_gan):
        G_losses = {}
        pcloud = data['pnts'].float().cuda()
        normals = data['normals'].float().cuda()
        img = data['img'].float().cuda()
        seg = data['seg'].float().cuda()
        sample_coord = data['sample_coord'].float().cuda()
        sample_sdf = data['sample_sdf'].float().cuda()

        tensor = torch.concat([img,seg],dim=1)
        features = self.Encoder(tensor)
        #print(features.shape)
        #features = self.Encoder(img, pcloud)
        bs, _, d, h, w =seg.shape
        generated_sdf = self.Decoder(sample_coord, features)
        #generated_grid = self.Decoder(grid_coord, features)

        #pred_seg = rearrange(generated_grid,'bs (d h w) -> bs 1 d h w',d=d,h=h,w=w)
        #G_losses['seg_loss'] = self.criterionSeg(generated_sdf, sample_occ) + self.criterionSeg(pred_seg, seg) + \
        #        dice_loss(F.sigmoid(pred_seg), seg.float(),multiclass=False)
        #G_losses['seg_loss'] = self.criterionSeg(generated_sdf, sample_occ) + \
        #        dice_loss(F.sigmoid(pred_seg), seg.float(),multiclass=False)
        G_losses['seg_loss'] = self.criterionSeg(generated_sdf, sample_sdf)
        '''
        pnts = pcloud.clone()
        if len(pnts.shape) == 2:
            pnts = pnts.unsqueeze(0)
        nonmnfld_pnts = self.sampler.get_points(pnts, None)
        pnts.requires_grad_()
        nonmnfld_pnts.requires_grad_()
        
        mnfld_pred = self.Decoder(pnts, features)
        nonmnfld_pred = self.Decoder(nonmnfld_pnts, features)

        ## compute gradient
        mnfld_gradient = gradient(pnts, mnfld_pred)
        nonmnfld_gradient = gradient(nonmnfld_pnts, nonmnfld_pred)
        
        ## manifold loss
        #G_losses['mnf_loss'] = (mnfld_pred.abs()).mean()
        ## eikonal_loss
        G_losses['grad_loss'] = ((nonmnfld_gradient.norm(2, dim=-1) - 3) ** 2).mean()
        
        if normals is not None:
            G_losses['normal_loss'] = ((mnfld_gradient - normals).abs()).norm(2, dim=2).mean()
        '''
        #G_losses['rec_loss'] = self.criterionRec(generated_sdf, sample_sdf)
        #sample_occ = rearrange(sample_occ,'b (d h w) -> b 1 d h w',d=80,h=80,w=80)
        #generated_sdf = rearrange(generated_sdf,'b (d h w) -> b 1 d h w',d=80,h=80,w=80)

        if self.opt.latent_code_regularization:
            G_losses['latent_loss'] = self.ll_loss(features) 
        return G_losses, generated_sdf
    
    def run_generator_one_step(self, data, use_gan):
        self.optimizer_G.zero_grad()
        self.optimizer_E.zero_grad()
        g_losses, generated = self.compute_generator_loss(data, use_gan)
        print(g_losses)
        g_loss = g_losses['seg_loss'] + self.opt.lambda_ll * g_losses['latent_loss']
        #g_loss = g_losses['rec_loss'] + 0.1 * g_losses['grad_loss'] + g_losses['normal_loss'] + self.opt.lambda_ll * g_losses['latent_loss']
        self.scaler.scale(g_loss).backward()
        self.scaler.step(self.optimizer_E)
        self.scaler.step(self.optimizer_G)
        self.scaler.update()
        return g_losses, generated

    def run_evaluation_during_trainingEX(self, dataloader, grid_size):
        num_val = len(dataloader)
        self.eval()
        batch = None
        generated_sdf = None
        val_rec_loss = 0
        val_score = 0
        d, h ,w = grid_size
        with torch.no_grad():
            for _, batch in enumerate(dataloader):
                pcloud = batch['pnts'].float().cuda()
                img = batch['img'].float().cuda()
                seg = batch['seg'].float().cuda()
                coords = batch['sample_coord'].float().cuda()
                sample_sdf = batch['sample_sdf'].float().cuda()
                
                grid_coord = batch['grid_coord'].float().cuda()
                grid_occ = batch['grid_occ'].float().cuda()
                
                tensor = torch.concat([img,seg],dim=1)
                #grid_coord = batch['grid_coord'].float().cuda()
                #features = self.Encoder(img, pcloud)
                features = self.Encoder(tensor)
                generated_sdf = self.Decoder(coords, features)
                #generated_grid = self.Decoder(grid_coord, features)
                #pred_seg = rearrange(generated_grid,'bs (d h w) -> bs 1 d h w',d=d,h=h,w=w)
                loss  = self.criterionSeg(generated_sdf, sample_sdf)
                
                generated_occ = self.Decoder(grid_coord, features)
                generated_occ = rearrange(generated_occ,'bs (d h w)-> bs 1 d h w',d=d,h=h,w=w)
                grid_occ = rearrange(grid_occ, 'bs (d h w) -> bs 1 d h w',d=d,h=h,w=w)
                val_score += dice_coeff3D((generated_occ<0).float(), grid_occ)
                val_rec_loss += loss
            xyz_coords = np.mgrid[:d, :h, :w].astype(np.float32)
            xyz_coords[0, ...] = xyz_coords[0, ...] / (d - 1)
            xyz_coords[1, ...] = xyz_coords[1, ...] / (h - 1)
            xyz_coords[2, ...] = xyz_coords[2, ...] / (w - 1)
            xyz_coords = (xyz_coords - 0.5) * 2
            xyz_coords = rearrange(xyz_coords, 'c d h w -> (d h w) c')
            xyz_coords = torch.from_numpy(xyz_coords)
            xyz_coords = xyz_coords.unsqueeze(0).cuda()
            #print(features[0].shape)
            generated_sdf = self.Decoder(xyz_coords, features[0].unsqueeze(0))
            generated_seg = generated_sdf < 0
            generated_seg = generated_seg.float()
            seg_gt = batch['seg'][0].float()
            img_gt = batch['img'][0].float()
        self.train()
        generated_seg = rearrange(generated_seg, '1 (d h w) ->d h w',d=d,h=h,w=w)
        generated_sdf = rearrange(generated_sdf, '1 (d h w) ->d h w',d=d,h=h,w=w)
        return seg_gt, img_gt, generated_seg, generated_sdf, val_rec_loss / num_val, val_score / num_val

    def update_learning_rate(self, epoch):
        #self.lr_scheduler_G.step()
        #if self.opt.use_gan and self.lr_scheduler_D is not None:
        #    self.lr_scheduler_D.step()
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
