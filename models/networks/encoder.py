"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from pickle import encode_long
from scipy.special import basic
from torch.jit import strict_fusion
from torch_scatter import scatter_max, scatter_mean
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.networks.attention import TransformerEncoder
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.unet import DoubleConv, UNet3D, UNet3DNeo, number_of_features_per_level, Encoder, Decoder, DecoderNeo
from models.networks.convlstm import INRConv3DLSTM,Conv3DLSTM
from util.util import pad_nd_image
from math import pi
class InitWeights_me(object):
    def __init__(self, init_type='normal', gain=0.02,neg_slope=1e-2):
        self.init_type = init_type
        self.gain = gain
        self.neg_slope = neg_slope

    def __call__(self,module):
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
            nn.init.normal_(module.weight, 1.0, self.gain)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear) \
                or isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
            if self.init_type == 'normal':
                nn.init.normal_(module.weight, 0.0, self.gain)
            elif self.init_type == 'xavier':
                nn.init.xavier_normal_(module.weight, self.gain)
            elif self.init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(module.weight, self.gain)
            elif self.init_type == 'kaiming':
                nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in')
            elif self.init_type =='orthogonal':
                nn.init.orthogonal_(module.weight, gain=self.gain)
            elif self.init_type == 'None':
                module.reset_parameters()
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % self.init_type)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

class Bottleneck3d(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck3d, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        #self.bn1 = nn.InstanceNorm3d(planes, affine=True)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #self.bn2 = nn.InstanceNorm3d(planes, affine=True)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, self.expansion*planes, kernel_size=1, bias=False)
        #self.bn3 = nn.InstanceNorm3d(self.expansion*planes, affine=True)
        self.bn3 = nn.BatchNorm3d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
               # nn.InstanceNorm3d(self.expansion*planes, affine=True)
                nn.BatchNorm3d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Res3DEncoderLSTM(BaseNetwork):
    "multi scale encoder inspired by fpn fully resolution"
    def __init__(self, num_in,  block, num_blocks ,ndf=64, kernel_size=3):
        super(Res3DEncoderLSTM, self).__init__()
        self.in_planes = ndf
        self.block_expansion = 4
        self.conv1 = nn.Conv3d(num_in, ndf, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.InstanceNorm3d(ndf, affine=True)

        ## Bottom-up layers
        self.layer1 = self._make_layer(block, ndf, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, ndf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, ndf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, ndf * 8, num_blocks[3], stride=2)

        self.toplayer = nn.Conv3d(ndf * 8 * self.block_expansion, 32, kernel_size=1, stride=1,padding=0)   #Reduce channels
        
        # Smooth layers
        self.smooth1 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)
        
        ## lateral layers
        self.latlayers1 = nn.Conv3d(ndf * 16, 32, kernel_size=1, stride=1,padding=0)
        self.latlayers2 = nn.Conv3d(ndf * 8, 32, kernel_size=1,stride=1,padding=0)
        self.latlayers3 = nn.Conv3d(ndf * 4, 32, kernel_size=1,stride=1,padding=0)
        
        self.convlstm = INRConv3DLSTM(input_dim=33,hidden_dim=[32],kernel_size=[(3,3,3)],num_layers=1,\
                batch_first=True,bias=True,return_all_layers=False)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        ''' upsample and add two feature maps'''
        _, _, D, H, W = y.size()
        return F.upsample(x, size=(D,H,W), mode='trilinear') + y

    def forward(self, x, time_interval):
        bs,s,_,d,h,w = x.shape
        x = rearrange(x, 'bs s c d h w -> (bs s) c d h w')
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        # Top-down & smooth
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayers1(c4))
        p4 = self.smooth1(p4)
        p3 = self._upsample_add(p4, self.latlayers2(c3))
        p3 = self.smooth2(p3)
        p2 = self._upsample_add(p3, self.latlayers3(c2))
        p2 = self.smooth3(p2)

        ## features from image
        fea = rearrange(p2,'(bs s) c d h w -> bs s c d h w',bs=bs,s=s)
        
        if time_interval is not None:
            time_interval = rearrange(time_interval,'bs s->bs s 1 1 1 1')
            time_interval = repeat(time_interval, 'bs s 1 1 1 1->bs s 1 80 80 80')
            fea_lstm = torch.concat([fea,time_interval],dim=2)
            fea_lstm = self.convlstm(fea_lstm,None)
        else:
            fea_lstm = self.convlstm(fea,None)

        return fea, fea_lstm
        #return p2

class Res3DEncoder(BaseNetwork):
    "multi scale encoder inspired by fpn fully resolution"
    def __init__(self, num_in, num_out, block, num_blocks ,ndf=64, kernel_size=3):
        super(Res3DEncoder, self).__init__()
        self.in_planes = ndf
        self.block_expansion = 4
        self.conv1 = nn.Conv3d(num_in, ndf, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(ndf, affine=True)

        ## Bottom-up layers
        self.layer1 = self._make_layer(block, ndf, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, ndf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, ndf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, ndf * 8, num_blocks[3], stride=2)

        self.toplayer = nn.Conv3d(ndf * 8 * self.block_expansion, num_out, kernel_size=1, stride=1,padding=0)   #Reduce channels
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        ''' upsample and add two feature maps'''
        _, _, D, H, W = y.size()
        return F.upsample(x, size=(D,H,W), mode='trilinear') + y

    def forward(self,x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        top = self.toplayer(c5)
        return top

class ConesEncoder(nn.Module):
    "multi scale encoder inspired by fpn fully resolution"
    def __init__(self, num_in,  block, num_blocks ,ndf=64, kernel_size=3):
        super(ConesEncoder, self).__init__()
        self.in_planes = ndf
        self.block_expansion = 4
        self.conv1 = nn.Conv2d(num_in, ndf, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.InstanceNorm2d(ndf, affine=True)

        ## Bottom-up layers
        self.layer1 = self._make_layer(block, ndf, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, ndf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, ndf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, ndf * 8, num_blocks[3], stride=2)

        self.toplayer = nn.Conv2d(ndf * 8 * self.block_expansion, 256, kernel_size=1, stride=1,padding=0)   #Reduce channels
        
        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        ## lateral layers
        self.latlayers1 = nn.Conv2d(ndf * 16, 256, kernel_size=1, stride=1,padding=0)
        self.latlayers2 = nn.Conv2d(ndf * 8, 256, kernel_size=1,stride=1,padding=0)
        self.latlayers3 = nn.Conv2d(ndf * 4, 256, kernel_size=1,stride=1,padding=0)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        ''' upsample and add two feature maps'''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self,x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        # Top-down & smooth
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayers1(c4))
        p4 = self.smooth1(p4)
        p3 = self._upsample_add(p4, self.latlayers2(c3))
        p3 = self.smooth2(p3)
        p2 = self._upsample_add(p3, self.latlayers3(c2))
        p2 = self.smooth3(p2)
        return p2

class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx

def coordinate2index(x, reso, coord_type='2d'):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        x (tensor): coordinate
        reso (int): defined resolution
        coord_type (str): coordinate type
    '''
    x = (x * reso).long()
    if coord_type == '2d': # plane
        index = x[:, :, 0] + reso * x[:, :, 1]
    elif coord_type == '3d': # grid
        index = x[:, :, 0] + reso * (x[:, :, 1] + reso * x[:, :, 2])
    index = index[:, None, :]
    return index

def normalize_3d_coordinate(p, padding=0.1):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''
    
    p_nor = p / (1 + padding + 10e-4) # (-0.5, 0.5)
    p_nor = p_nor + 0.5 # range (0, 1)
    # f there are outliers out of the range
    if p_nor.max() >= 1:
        p_nor[p_nor >= 1] = 1 - 10e-4
    if p_nor.min() < 0:
        p_nor[p_nor < 0] = 0.0
    return p_nor

class TensorEncoderLSTMEX(BaseNetwork):
    def __init__(self, num_in=3, dim=32, hidden_dim=32, n_blocks=5):
        super().__init__()
        self.num_in = num_in
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        self.conv_in = nn.Conv3d(in_channels=2, out_channels=32,kernel_size=3,stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        #self.conv_mid = nn.Conv3d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding=1)
        
        self.reso_grid = 80
        self.unet3d = UNet3D(in_channels=self.dim, out_channels=self.dim, f_maps=self.hidden_dim, num_levels=4)
        
        self.convlstm = INRConv3DLSTM(input_dim=33,hidden_dim=[32],kernel_size=[(3,3,3)],num_layers=1,\
                batch_first=True,bias=True,return_all_layers=False)
        #self.convlstm = INRConv3DLSTM(input_dim=33,hidden_dim=[32,32,32],kernel_size=[(3,3,3),(3,3,3),(3,3,3)],num_layers=3,\
        #        batch_first=True,bias=True,return_all_layers=False)
        self.init = InitWeights_me('xavier', 0.02)
        self.apply(self.init)

    def forward(self, img, time_interval):
        bs,s,_,d,h,w = img.shape
        ## features from image
        img = rearrange(img, 'bs s c d h w -> (bs s) c d h w')
        
        fea = self.conv_in(img)
        fea = F.relu(self.bn1(fea))
        fea = self.unet3d(fea)
        
        fea = rearrange(fea,'(bs s) c d h w -> bs s c d h w',bs=bs,s=s)
        
        if time_interval is not None:
            time_interval = rearrange(time_interval,'bs s->bs s 1 1 1 1')
            time_interval = repeat(time_interval, 'bs s 1 1 1 1->bs s 1 80 80 80')
            fea_lstm = torch.concat([fea,time_interval],dim=2)
            fea_lstm = self.convlstm(fea_lstm,None)
        else:
            fea_lstm = self.convlstm(fea,None)

        return fea, fea_lstm

class TensorLSTM(BaseNetwork):
    def __init__(self, num_in=3, hidden_dim=32):
        super().__init__()
        self.num_in = num_in
        self.hidden_dim = hidden_dim
        self.convlstm = INRConv3DLSTM(input_dim=self.num_in+1,hidden_dim=[self.hidden_dim,self.hidden_dim,self.hidden_dim],kernel_size=[(3,3,3),(3,3,3),(3,3,3)],num_layers=3,\
                batch_first=True,bias=True,return_all_layers=False)
        
        self.init = InitWeights_me('xavier', 0.02)
        self.apply(self.init)

    def forward(self, fea, time_interval):
        bs,s,_,d,h,w = fea.shape
        #fea = rearrange(fea,'(bs s) c d h w -> bs s c d h w',bs=bs,s=s)
        
        if time_interval is not None:
            time_interval = rearrange(time_interval,'bs s->bs s 1 1 1 1')
            time_interval = repeat(time_interval, 'bs s 1 1 1 1->bs s 1 d h w',d=d,h=h,w=w)
            fea_lstm = torch.concat([fea,time_interval],dim=2)
            fea_lstm = self.convlstm(fea_lstm,None)
        else:
            fea_lstm = self.convlstm(fea,None)

        return fea_lstm


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class ViTEncoder(BaseNetwork):
    def __init__(self, image_size, image_patch_size, frames, frame_patch_size, num_classes, \
            dim, depth, heads, mlp_dim, channels = 2, dim_head = 64, dropout = 0.1, emb_dropout = 0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'
        self.patch_size = image_patch_size
        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()
            
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.init = InitWeights_me('xavier', 0.02)
        self.apply(self.init)
        torch.nn.init.xavier_normal_(self.cls_token,gain=0.02)
        torch.nn.init.xavier_normal_(self.pos_embedding,gain=0.02)

    def forward(self, video):
        x = self.to_patch_embedding(video)
        b, n, _ = x.shape
        #cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        #x = torch.cat((x,cls_tokens), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)
        x = x[:,-1,:]
        x = self.to_latent(x)
        x = self.mlp_head(x)
        x = x.unsqueeze(1)
        #print(x.shape)
        #x = rearrange(x,'b (d h w) c -> b c d h w',d=self.patch_size,h=self.patch_size,w=self.patch_size)
        return x

class TensorDownX4EncoderLSTMold(BaseNetwork):
    ##### downsample x2
    def __init__(self, num_in=3, dim=32, hidden_dim=32, n_blocks=5):
        super().__init__()
        self.num_in = num_in
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.conv_in = nn.Conv3d(in_channels=2, out_channels=32,kernel_size=3,stride=2, padding=1)
        
        self.unet3d = UNet3D(in_channels=self.dim, out_channels=self.dim, f_maps=self.hidden_dim, num_levels=4)
        self.conv_out = nn.Conv3d(in_channels=32,out_channels=32,kernel_size=3,stride=2,padding=1)
        
        self.convlstm = INRConv3DLSTM(input_dim=33,hidden_dim=[64,64,32],kernel_size=[(3,3,3),(3,3,3),(3,3,3)], num_layers=3,\
                batch_first=True,bias=True,return_all_layers=False)
        
        self.init = InitWeights_me('xavier', 0.02)
        self.apply(self.init)
    
    def forward(self, img, time_interval):
        bs,s,_,d,h,w = img.shape
        ## features from image
        img = rearrange(img, 'bs s c d h w -> (bs s) c d h w')
        ## features from image
        fea = self.conv_in(img)
        fea = self.unet3d(fea)
        fea = self.conv_out(fea)
        fea = rearrange(fea,'(bs s) c d h w -> bs s c d h w',bs=bs,s=s)
        time_interval = rearrange(time_interval,'bs s ->bs s 1 1 1 1')
        time_interval = repeat(time_interval, 'bs s c 1 1 1->bs s c 16 16 16')
        
        fea_lstm = torch.concat([fea[:,0:2,:], time_interval],dim=2)
        fea_lstm = self.convlstm(fea_lstm,None)

        return fea, fea_lstm

class TensorDownX4EncoderLSTMNot(BaseNetwork):
    ##### downsample x2
    def __init__(self, num_in=3, dim=32, hidden_dim=32, n_blocks=5):
        super().__init__()
        self.num_in = num_in
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.conv_in = nn.Conv3d(in_channels=2, out_channels=32,kernel_size=3,stride=2, padding=1)
        
        self.unet3d = UNet3D(in_channels=self.dim, out_channels=self.dim, f_maps=self.hidden_dim, num_levels=4)
        self.conv_out = nn.Conv3d(in_channels=32,out_channels=32,kernel_size=3,stride=2,padding=1)
        self.convlstm = INRConv3DLSTM(input_dim=32,hidden_dim=[64,64,32],kernel_size=[(3,3,3),(3,3,3),(3,3,3)], num_layers=3,\
                batch_first=True,bias=True,return_all_layers=False)
        self.init = InitWeights_me('xavier', 0.02)
        self.apply(self.init)

    def forward(self, img, time_interval):
        bs,s,_,d,h,w = img.shape
        ## features from image
        img = rearrange(img, 'bs s c d h w -> (bs s) c d h w')
        ## features from image
        fea = self.conv_in(img)
        fea = self.unet3d(fea)
        fea = self.conv_out(fea)
        fea = rearrange(fea,'(bs s) c d h w -> bs s c d h w',bs=bs,s=s)
        fea_lstm = self.convlstm(fea[:,0:2,:],None)

        return fea, fea_lstm

class TensorDownX4EncoderLSTM(BaseNetwork):
    ##### downsample x2
    def __init__(self, num_in=3, dim=32, hidden_dim=32, n_blocks=5,order=0):
        super().__init__()
        self.num_in = num_in
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.conv_in = nn.Conv3d(in_channels=2, out_channels=32,kernel_size=3,stride=2, padding=1)
        
        self.unet3d = UNet3D(in_channels=self.dim, out_channels=self.dim, f_maps=self.hidden_dim, num_levels=4)
        self.conv_out = nn.Conv3d(in_channels=32,out_channels=32,kernel_size=3,stride=2,padding=1)
        
        self.order = order
        print('order:', order + 1)
        self.dropout = nn.Dropout3d(p=0.5) 
        self.convlstm = INRConv3DLSTM(input_dim=32 + (order + 1) * 2,hidden_dim=[32,32,32],kernel_size=[(3,3,3),(3,3,3),(3,3,3)], num_layers=3,\
                batch_first=True,bias=True,return_all_layers=False)
        
        self.init = InitWeights_me('xavier', 0.02)
        self.apply(self.init)

    def time_coding(self,t, ds):
        """Creates the position encoding for the pixel-wise MLPs"""
        f0 = 0
        #print(t)
        while f0 <= ds:
            f = pow(2, f0)
            xcos = torch.cos(f * pi * t.float())
            xsin = torch.sin(f * pi * t.float())
            #print(xcos.shape, xcos)
            coords_cur = torch.cat([xcos.unsqueeze(2), xsin.unsqueeze(2)], 2)
            if f0 == 0:
                coords = coords_cur
            else:
                coords = torch.cat([coords, coords_cur], 2)
            #print('coords', coords.shape)
            f0 = f0 + 1
        return coords
    
    def forward(self, img, time_interval):
        bs,s,_,d,h,w = img.shape
        ## features from image
        img = rearrange(img, 'bs s c d h w -> (bs s) c d h w')
        ## features from image
        fea = self.conv_in(img)
        fea = self.unet3d(fea)
        fea = self.conv_out(fea)
        fea = rearrange(fea,'(bs s) c d h w -> bs s c d h w',bs=bs,s=s)
        #print(time_interval)
        encoded_times = self.time_coding(time_interval,self.order) 
        encoded_times = rearrange(encoded_times,'bs s c ->bs s c 1 1 1')
        encoded_times = repeat(encoded_times, 'bs s c 1 1 1->bs s c 16 16 16')
        encoded_times = rearrange(encoded_times,'bs s c d h w -> (bs s) c d h w')
        encoded_times = self.dropout(encoded_times)
        #fea_lstm = self.conv_time(fea_lstm)
        encoded_times = rearrange(encoded_times,'(bs s) c d h w -> bs s c d h w',bs=bs,s=2)
        
        fea_lstm = torch.concat([fea[:,0:2,:],encoded_times],dim=2)
        fea_lstm = self.convlstm(fea_lstm,None)

        return fea, fea_lstm

class TensorDownX8EncoderLSTM(BaseNetwork):
    ##### downsample x8
    def __init__(self, num_in=3, dim=32, hidden_dim=32, n_blocks=5,order=5):
        super().__init__()
        self.num_in = num_in
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.conv_in = nn.Conv3d(in_channels=2, out_channels=32,kernel_size=3,stride=2, padding=1)
        
        self.unet3d = UNet3DNeo(in_channels=self.dim, out_channels=self.dim, f_maps=self.hidden_dim, num_levels=4)
        self.pooling = nn.MaxPool3d(kernel_size=3,stride=2)
        self.conv_out = nn.Conv3d(in_channels=32,out_channels=32,kernel_size=3,stride=2,padding=1)
        
        self.order = order
        print('order:', order + 1)
        #self.conv_time = nn.Conv3d(in_channels=32 + (order + 1) * 2, out_channels=32,kernel_size=1,stride=1,padding=0)
        self.dropout = nn.Dropout3d(p=0.5) 
        self.convlstm = INRConv3DLSTM(input_dim=32 + (order + 1) * 2,hidden_dim=[64,64,32],kernel_size=[(3,3,3),(3,3,3),(3,3,3)], num_layers=3,\
                batch_first=True,bias=True,return_all_layers=False)
        self.init = InitWeights_me('xavier', 0.02)
        self.apply(self.init)
    
    def time_coding(self,t, ds):
        """Creates the position encoding for the pixel-wise MLPs"""
        f0 = 0
        #print(t)
        while f0 <= ds:
            f = pow(2, f0)
            xcos = torch.cos(f * pi * t.float())
            xsin = torch.sin(f * pi * t.float())
            #print(xcos.shape, xcos)
            coords_cur = torch.cat([xcos.unsqueeze(2), xsin.unsqueeze(2)], 2)
            if f0 == 0:
                coords = coords_cur
            else:
                coords = torch.cat([coords, coords_cur], 2)
            #print('coords', coords.shape)
            f0 = f0 + 1
        return coords

    def forward(self, img, time_interval):
        bs,s,_,d,h,w = img.shape
        ## features from image
        img = rearrange(img, 'bs s c d h w -> (bs s) c d h w')
        
        fea = self.conv_in(img)
        fea = self.unet3d(fea)
        fea = self.pooling(fea)
        fea = self.conv_out(fea)
        fea = rearrange(fea,'(bs s) c d h w -> bs s c d h w',bs=bs,s=s)
        #print(time_interval)
        encoded_times = self.time_coding(time_interval,self.order) 
        encoded_times = rearrange(encoded_times,'bs s c ->bs s c 1 1 1')
        encoded_times = repeat(encoded_times, 'bs s c 1 1 1->bs s c 8 8 8')
        encoded_times = rearrange(encoded_times,'bs s c d h w -> (bs s) c d h w')
        encoded_times = self.dropout(encoded_times)
        #fea_lstm = self.conv_time(fea_lstm)
        encoded_times = rearrange(encoded_times,'(bs s) c d h w -> bs s c d h w',bs=bs,s=2)
        
        fea_lstm = torch.concat([fea[:,0:2,:],encoded_times],dim=2)
        fea_lstm = self.convlstm(fea_lstm,None)

        return fea, fea_lstm

class TensorDownX8Encoder(BaseNetwork):
    ##### downsample x4
    def __init__(self, num_in=3, dim=32, hidden_dim=32, n_blocks=5):
        super().__init__()
        self.num_in = num_in
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.conv_in = nn.Conv3d(in_channels=2, out_channels=32,kernel_size=3,stride=2, padding=1)
        
        self.unet3d = UNet3DNeo(in_channels=self.dim, out_channels=self.dim, f_maps=self.hidden_dim, num_levels=4)
        self.pooling = nn.MaxPool3d(kernel_size=3,stride=2)
        self.conv_out = nn.Conv3d(in_channels=32,out_channels=32,kernel_size=3,stride=2,padding=1)
        self.init = InitWeights_me('xavier', 0.02)
        self.apply(self.init)

    def forward(self, img):
        fea = self.conv_in(img)
        fea = self.unet3d(fea)
        fea = self.pooling(fea)
        fea = self.conv_out(fea)
        return fea

class TensorDownX4Encoder(BaseNetwork):
    ##### downsample x4
    def __init__(self, num_in=3, dim=32, hidden_dim=32, n_blocks=5):
        super().__init__()
        self.num_in = num_in
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.conv_in = nn.Conv3d(in_channels=2, out_channels=32,kernel_size=3,stride=2, padding=1)
        
        self.unet3d = UNet3DNeo(in_channels=self.dim, out_channels=self.dim, f_maps=self.hidden_dim, num_levels=4)
        self.conv_out = nn.Conv3d(in_channels=32,out_channels=32,kernel_size=3,stride=2,padding=1)
        self.init = InitWeights_me('xavier', 0.02)
        self.apply(self.init)

    def forward(self, img):
        fea = self.conv_in(img)
        fea = self.unet3d(fea)
        fea = self.conv_out(fea)
        return fea

class TensorDownEncoderLSTMDiff(BaseNetwork):
    ##### downsample x2
    def __init__(self, num_in=3, dim=32, hidden_dim=32, n_blocks=5):
        super().__init__()
        self.num_in = num_in
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.conv_in = nn.Conv3d(in_channels=2, out_channels=32,kernel_size=3,stride=2, padding=1)
        
        self.unet3d = UNet3D(in_channels=self.dim, out_channels=self.dim, f_maps=self.hidden_dim, num_levels=4)
        self.convlstm = INRConv3DLSTM(input_dim=33,hidden_dim=[32,32,32],kernel_size=[(3,3,3),(3,3,3),(3,3,3)], num_layers=3,\
                batch_first=True,bias=True,return_all_layers=False)
        self.init = InitWeights_me('xavier', 0.02)
        self.apply(self.init)

    def forward(self, img, time_interval):
        bs,s,_,d,h,w = img.shape
        ## features from image
        img = rearrange(img, 'bs s c d h w -> (bs s) c d h w')
        ## features from image
        fea = self.conv_in(img)
        fea = self.unet3d(fea)
        fea = rearrange(fea,'(bs s) c d h w -> bs s c d h w',bs=bs,s=s)
        
        if time_interval is not None:
            time_interval = rearrange(time_interval,'bs s->bs s 1 1 1 1')
            time_interval = repeat(time_interval, 'bs s 1 1 1 1->bs s 1 32 32 32')
            fea_lstm = torch.concat([fea,time_interval],dim=2)
            fea_lstm = self.convlstm(fea_lstm)
        else:
            fea_lstm = self.convlstm(fea)

        return fea, fea_lstm

class TensorDownEncoderLSTMET(BaseNetwork):
    ##### downsample x2 with encoded time
    def __init__(self, num_in=3, dim=32, hidden_dim=32, n_blocks=5,order=5):
        super().__init__()
        self.num_in = num_in
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.conv_in = nn.Conv3d(in_channels=2, out_channels=32,kernel_size=3,stride=2, padding=1)
        
        self.unet3d = UNet3D(in_channels=self.dim, out_channels=self.dim, f_maps=self.hidden_dim, num_levels=4)
        self.order = order
        print('order:', order + 1)
        self.dropout = nn.Dropout3d(p=0.5) 
        self.convlstm = INRConv3DLSTM(input_dim=32 + (self.order + 1) * 2,hidden_dim=[32,32,32],kernel_size=[(3,3,3),(3,3,3),(3,3,3)], num_layers=3,\
                batch_first=True,bias=True,return_all_layers=False)
        self.init = InitWeights_me('xavier', 0.02)
        self.apply(self.init)
        print('time encoding factor', self.order)

    def time_coding(self,t, ds):
        """Creates the position encoding for the pixel-wise MLPs"""
        f0 = 0
        #print(t)
        while f0 <= ds:
            f = pow(2, f0)
            xcos = torch.cos(f * pi * t.float())
            xsin = torch.sin(f * pi * t.float())
            #print(xcos.shape, xcos)
            coords_cur = torch.cat([xcos.unsqueeze(2), xsin.unsqueeze(2)], 2)
            if f0 == 0:
                coords = coords_cur
            else:
                coords = torch.cat([coords, coords_cur], 2)
            #print('coords', coords.shape)
            f0 = f0 + 1
        return coords
    
    def forward(self, img, time_interval):
        bs,s,_,d,h,w = img.shape
        ## features from image
        img = rearrange(img, 'bs s c d h w -> (bs s) c d h w')
        ## features from image
        fea = self.conv_in(img)
        fea = self.unet3d(fea)
        fea = rearrange(fea,'(bs s) c d h w -> bs s c d h w',bs=bs,s=s)
        #print(time_interval)
        encoded_times = self.time_coding(time_interval,self.order) 
        encoded_times = rearrange(encoded_times,'bs s c ->bs s c 1 1 1')
        encoded_times = repeat(encoded_times, 'bs s c 1 1 1->bs s c 32 32 32')
        encoded_times = rearrange(encoded_times,'bs s c d h w -> (bs s) c d h w')
        encoded_times = self.dropout(encoded_times)
        encoded_times = rearrange(encoded_times,'(bs s) c d h w -> bs s c d h w',bs=bs,s=2)
        
        fea_lstm = torch.concat([fea[:,0:2,:],encoded_times],dim=2)
        fea_lstm = self.convlstm(fea_lstm,None)
        

        return fea, fea_lstm

class TensorDownEncoderLSTMNot(BaseNetwork):
    ##### downsample x2
    def __init__(self, num_in=3, dim=32, hidden_dim=32, n_blocks=5):
        super().__init__()
        self.num_in = num_in
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.conv_in = nn.Conv3d(in_channels=2, out_channels=32,kernel_size=3,stride=2, padding=1)
        
        self.unet3d = UNet3D(in_channels=self.dim, out_channels=self.dim, f_maps=self.hidden_dim, num_levels=4)
        self.convlstm = INRConv3DLSTM(input_dim=32,hidden_dim=[32,32,32],kernel_size=[(3,3,3),(3,3,3),(3,3,3)], num_layers=3,\
                batch_first=True,bias=True,return_all_layers=False)
        self.init = InitWeights_me('xavier', 0.02)
        self.apply(self.init)

    def forward(self, img, time_interval):
        bs,s,_,d,h,w = img.shape
        ## features from image
        img = rearrange(img, 'bs s c d h w -> (bs s) c d h w')
        ## features from image
        fea = self.conv_in(img)
        fea = self.unet3d(fea)
        fea = rearrange(fea,'(bs s) c d h w -> bs s c d h w',bs=bs,s=s)
        
        fea_lstm = self.convlstm(fea[:,0:2,:])

        return fea, fea_lstm

class TensorDownEncoderLSTM(BaseNetwork):
    ##### downsample x2
    def __init__(self, num_in=3, dim=32, hidden_dim=32, n_blocks=5):
        super().__init__()
        self.num_in = num_in
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.conv_in = nn.Conv3d(in_channels=2, out_channels=32,kernel_size=3,stride=2, padding=1)
        
        self.unet3d = UNet3D(in_channels=self.dim, out_channels=self.dim, f_maps=self.hidden_dim, num_levels=4)
        self.convlstm = INRConv3DLSTM(input_dim=33,hidden_dim=[32,32,32],kernel_size=[(3,3,3),(3,3,3),(3,3,3)], num_layers=3,\
                batch_first=True,bias=True,return_all_layers=False)
        self.init = InitWeights_me('xavier', 0.02)
        self.apply(self.init)

    def forward(self, img, time_interval):
        bs,s,_,d,h,w = img.shape
        ## features from image
        img = rearrange(img, 'bs s c d h w -> (bs s) c d h w')
        ## features from image
        fea = self.conv_in(img)
        fea = self.unet3d(fea)
        fea = rearrange(fea,'(bs s) c d h w -> bs s c d h w',bs=bs,s=s)
        
        if time_interval is not None:
            time_interval = rearrange(time_interval,'bs s->bs s 1 1 1 1')
            time_interval = repeat(time_interval, 'bs s 1 1 1 1->bs s 1 32 32 32')
            fea_lstm = torch.concat([fea[:,0:2,:],time_interval],dim=2)
            fea_lstm = self.convlstm(fea_lstm,None)
        else:
            fea_lstm = self.convlstm(fea[:,0:2,:],None)

        return fea, fea_lstm

class TensorDownEncoder(BaseNetwork):
    ##### downsample x2
    def __init__(self, num_in=3, dim=32, hidden_dim=32, n_blocks=5):
        super().__init__()
        self.num_in = num_in
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.conv_in = nn.Conv3d(in_channels=2, out_channels=32,kernel_size=3,stride=2, padding=1)
        
        self.unet3d = UNet3D(in_channels=self.dim, out_channels=self.dim, f_maps=self.hidden_dim, num_levels=4)
        self.init = InitWeights_me('xavier', 0.02)
        self.apply(self.init)

    def forward(self, img):
        ## features from image
        fea = self.conv_in(img)
        fea = self.unet3d(fea)
        return fea

class TensorEncoderLSTM(BaseNetwork):
    def __init__(self, num_in=3, dim=32, hidden_dim=32, n_blocks=5,order=5):
        super().__init__()
        self.num_in = num_in
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        self.conv_in = nn.Conv3d(in_channels=2, out_channels=32,kernel_size=3,stride=1, padding=1)
        self.unet3d = UNet3D(in_channels=self.dim, out_channels=self.dim, f_maps=self.hidden_dim, num_levels=4)
        
        self.order = order
        print('order:', order + 1)
        #self.conv_time = nn.Conv3d(in_channels=32 + (order + 1) * 2, out_channels=32,kernel_size=1,stride=1,padding=0)
        self.dropout = nn.Dropout3d(p=0.5) 
        self.convlstm = INRConv3DLSTM(input_dim=32 + (order + 1) * 2,hidden_dim=[32,32,32],kernel_size=[(3,3,3),(3,3,3),(3,3,3)], num_layers=3,\
                batch_first=True,bias=True,return_all_layers=False)
        
        self.init = InitWeights_me('xavier', 0.02)
        self.apply(self.init)
    
    def time_coding(self,t, ds):
        """Creates the position encoding for the pixel-wise MLPs"""
        f0 = 0
        #print(t)
        while f0 <= ds:
            f = pow(2, f0)
            xcos = torch.cos(f * pi * t.float())
            xsin = torch.sin(f * pi * t.float())
            #print(xcos.shape, xcos)
            coords_cur = torch.cat([xcos.unsqueeze(2), xsin.unsqueeze(2)], 2)
            if f0 == 0:
                coords = coords_cur
            else:
                coords = torch.cat([coords, coords_cur], 2)
            #print('coords', coords.shape)
            f0 = f0 + 1
        return coords

    def forward(self, img, time_interval):
        bs,s,_,d,h,w = img.shape
        ## features from image
        img = rearrange(img, 'bs s c d h w -> (bs s) c d h w')
        ## features from image
        fea = self.conv_in(img)
        fea = self.unet3d(fea)
        fea = rearrange(fea,'(bs s) c d h w -> bs s c d h w',bs=bs,s=s)
        #print(time_interval)
        encoded_times = self.time_coding(time_interval,self.order) 
        encoded_times = rearrange(encoded_times,'bs s c ->bs s c 1 1 1')
        encoded_times = repeat(encoded_times, 'bs s c 1 1 1->bs s c 64 64 64')
        encoded_times = rearrange(encoded_times,'bs s c d h w -> (bs s) c d h w')
        encoded_times = self.dropout(encoded_times)
        #fea_lstm = self.conv_time(fea_lstm)
        encoded_times = rearrange(encoded_times,'(bs s) c d h w -> bs s c d h w',bs=bs,s=2)
        
        fea_lstm = torch.concat([fea[:,0:2,:],encoded_times],dim=2)
        fea_lstm = self.convlstm(fea_lstm,None)

        return fea, fea_lstm

class TensorEncoder(BaseNetwork):
    def __init__(self, num_in=3, dim=32, hidden_dim=32, n_blocks=5):
        super().__init__()
        self.num_in = num_in
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        self.conv_in = nn.Conv3d(in_channels=2, out_channels=32,kernel_size=3,stride=1, padding=1)
        self.unet3d = UNet3D(in_channels=self.dim, out_channels=self.dim, f_maps=self.hidden_dim, num_levels=4)
        self.init = InitWeights_me('xavier', 0.02)
        self.apply(self.init)

    def forward(self, img):
        ## features from image
        fea = self.conv_in(img)
        fea = self.unet3d(fea)
        return fea

class MixEncoder(BaseNetwork):
    def __init__(self, num_in=3, dim=32, hidden_dim=32, n_blocks=5):
        super().__init__()
        self.num_in = num_in
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.fc_pos = nn.Linear(self.num_in, 2 * self.hidden_dim)
        
        self.conv_in = nn.Conv3d(in_channels=1, out_channels=32,kernel_size=3,stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv_mid = nn.Conv3d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding=1)
        
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2*hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        self.fc_out = nn.Linear(self.hidden_dim, self.dim)
        self.reso_grid=80
        self.unet3d = UNet3D(in_channels=self.dim, out_channels=self.dim, f_maps=self.hidden_dim, num_levels=4)
        #self.unet3d = UNet3DNeo(in_channels=self.dim, out_channels=self.dim, f_maps=self.hidden_dim, num_levels=4)
        self.scatter = scatter_max
        self.padding = 0.1
        self.init = InitWeights_me('xavier', 0.02)
        self.apply(self.init)
        
    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()
        c_out = 0
        for key in keys:
            # scatter plane features from points
            fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_grid**3)
            #print(fea[0].shape)
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

    def forward(self, img, p):
        bs, n, d = p.size()

        ## acquire the index for each point
        coord = {}
        index = {}
        
        ## features from image
        #img = rearrange(img, 'bs s d h w -> (bs s) 1 d h w')
        fea_img = self.conv_in(img)
        fea_img = F.relu(self.bn1(fea_img))

        coord['grid'] = normalize_3d_coordinate(p.clone(), padding=0.1)
        index['grid'] = coordinate2index(coord['grid'], 80, coord_type='3d')
        #print(index['grid'].shape)
        #print(index['grid'])
        net = self.fc_pos(p)
        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)
        c = self.fc_out(net)
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type='3d')
        
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.dim, self.reso_grid**3)
        #print(fea_grid.shape, fea_grid)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid) # B x C x reso^3
        #print(fea_grid.shape,fea_grid)
        fea_grid = fea_grid.reshape(p.size(0), self.dim, self.reso_grid, self.reso_grid, self.reso_grid) # sparce matrix (B x 512 x reso x reso)
        
        fea_mix = torch.concat([fea_grid,fea_img], dim=1)
        fea_mix = self.conv_mid(fea_mix)
        
        fea_mix = self.unet3d(fea_mix)
        return fea_mix

class PointEncoder(BaseNetwork):
    def __init__(self, num_in=3, dim=32, hidden_dim=32, n_blocks=5):
        super().__init__()
        self.num_in = num_in
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.fc_pos = nn.Linear(self.num_in, 2 * self.hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2*hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        self.fc_out = nn.Linear(self.hidden_dim, self.dim)
        self.reso_grid=80
        self.unet3d = UNet3D(in_channels=self.dim, out_channels=self.dim, f_maps=self.hidden_dim, num_levels=3)
        self.scatter = scatter_max
        self.padding = 0.1
        self.init = InitWeights_me('xavier', 0.02)
        self.apply(self.init)
        
    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_grid**3)
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

    def forward(self, p):
        bs, n, d = p.size()

        ## acquire the index for each point
        coord = {}
        index = {}
        coord['grid'] = normalize_3d_coordinate(p.clone(), padding=0.1)
        index['grid'] = coordinate2index(coord['grid'], 80, coord_type='3d')
        net = self.fc_pos(p)
        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)
        c = self.fc_out(net)
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type='3d')
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.dim, self.reso_grid**3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid) # B x C x reso^3
        fea_grid = fea_grid.reshape(p.size(0), self.dim, self.reso_grid, self.reso_grid, self.reso_grid) # sparce matrix (B x 512 x reso x reso)
        fea_grid = self.unet3d(fea_grid)
        return fea_grid

