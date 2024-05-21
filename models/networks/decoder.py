from random import sample
from einops import rearrange, repeat
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
#import torch as th
import torch
from math import pi
import numpy as np 
from util.util import InitWeights_me, grid_sample
import math
### define ######
OMEGA = 30

class positional_encoding(object):
    ''' Positional Encoding (presented in NeRF)

    Args:
        basis_function (str): basis function
    '''
    def __init__(self, basis_function='sin_cos'):
        super().__init__()
        self.func = basis_function

        L = 6
        freq_bands = 2.**(np.linspace(0, L-1, L))
        self.freq_bands = freq_bands * math.pi

    def __call__(self, p):
        if self.func == 'sin_cos':
            out = []
            #p = 2.0 * p - 1.0 # chagne to the range [-1, 1]
            for freq in self.freq_bands:
                out.append(torch.sin(freq * p))
                out.append(torch.cos(freq * p))
            p = torch.cat(out, dim=2)
        return p


class SirenActivation(nn.Module):
    def __init__(
        self
    ):
        super().__init__()

    def forward(self, input):
        return torch.sin(OMEGA * input)

    @staticmethod
    def sine_init(m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / OMEGA, np.sqrt(6 / num_input) / OMEGA)

    @staticmethod
    def first_layer_sine_init(m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-1 / num_input, 1 / num_input)

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

class NFCoordConcatDecoder3DMulti(BaseNetwork):
    """pixel-wise MLP with 3D(x,y,z) input"""
    def __init__(self, num_outputs=1, latent_dim=128, dims=[64,64,64,64], maximal_period=96, last_tanh=False, use_sine=False):
        super(NFCoordConcatDecoder3DMulti, self).__init__()

        self.num_outputs = num_outputs
        self.use_sine = use_sine
        self.num_params = latent_dim
        self.last_tanh = last_tanh
        self.channels = [3 + self.num_params] + dims + [num_outputs]
        latent_drop = False
        self.biases = []
        print(self.channels)
        self.net = nn.ModuleList()
        for i in range(len(self.channels) - 1):
            lin = nn.Linear(self.channels[i], self.channels[i+1], bias=True)
            if i == len(self.channels) - 2:
                torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi)/ np.sqrt(self.channels[i]), std=0.00001)
                torch.nn.init.constant_(lin.bias, -1)
            else:
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(self.channels[i+1]))
                torch.nn.init.constant_(lin.bias, 0.0)
            self.net.append(lin)
        if latent_drop:
            self.lat_drop = nn.Dropout(0.2)
        else:
            self.lat_drop = None
        #self.init = InitWeights_me('xavier', 0.02)
        #self.apply(self.init)

    def _set_num_params(self):
        nparams = 0
        # go over input/output channels for each layer
        idx = 0
        for layer, nci in enumerate(self.channels[:-1]):
            nco = self.channels[layer + 1] 
            nparams += nco  # FC biases
            self.biases.append((idx, idx + nco))
            idx += nco
        self.num_params = nparams

    def _get_bias_indices(self, idx):
        return self.biases[idx]

    def forward(self, coords, lr_params):
        bs, t, n, p = coords.shape
        out = rearrange(coords, 'bs t n p -> (bs t) n p')
        #coords = coords.reshape(bs, n, 1, 1, p)
        coords = rearrange(coords, 'bs t n p -> (bs t) n 1 1 p')
        lr_params = rearrange(lr_params, 'b t c d h w -> (b t) c d h w')
        sample_params = F.grid_sample(lr_params, coords, padding_mode='border', align_corners=True)
        sample_params = rearrange(sample_params,'b c n 1 1 -> b n c')
        out = torch.concat([out, sample_params], dim=2)
        num_layers = len(self.channels) - 1
        for idx, nci in enumerate(self.channels[:-1]):
            nco = self.channels[idx + 1]
            out = self.net[idx](out)
            # Apply RelU non-linearity in all but the last layer, and tanh in the last
            if idx < num_layers - 1:
                out = torch.nn.functional.leaky_relu(out, 0.01, inplace=True)
            #else:
            else:
                if self.last_tanh:
                    out = F.tanh(out)
                else:
                    out = out
        out = rearrange(out,'(b t) n 1 -> b t n', b=bs,t=t)
        return out

class NFCoordSirenModulationDecoder3D(BaseNetwork):
    """pixel-wise MLP with 3D(x,y,z) input"""
    def __init__(self, num_outputs=1, latent_dim=128, dims=[64,64,64,64], maximal_period=96, use_sine=True):
        super(NFCoordSirenModulationDecoder3D, self).__init__()

        self.num_outputs = num_outputs
        self.use_sine = use_sine
        self.num_params = 0   ## initialization
        self.channels = [3] + dims + [num_outputs]
        latent_drop = False
        self.biases = []
        print(self.channels)
        self._set_num_params()
        self.net = nn.ModuleList()
        self.sine = SirenActivation()
        for i in range(len(self.channels) - 1):
            lin = nn.Linear(self.channels[i], self.channels[i+1], bias=True)
            self.net.append(lin)
        if latent_drop:
            self.lat_drop = nn.Dropout(0.2)
        else:
            self.lat_drop = None
        
        if self.use_sine:
            print('now use siren')
            self.apply(self.sine.sine_init)
            self.net[0].apply(self.sine.first_layer_sine_init)

    def _set_num_params(self):
        nparams = 0
        # go over input/output channels for each layer
        idx = 0
        for layer, nci in enumerate(self.channels[:-1]):
            nco = self.channels[layer + 1] 
            nparams += nco  # FC biases
            self.biases.append((idx, idx + nco))
            idx += nco
        self.num_params = nparams

    def _get_bias_indices(self, idx):
        return self.biases[idx]

    def forward(self, coords, lr_params):
        bs, n, p = coords.shape
        out = coords
        coords = coords.reshape(bs, n, 1, 1, p)
        sample_params = F.grid_sample(lr_params, coords, padding_mode='border', align_corners=True)
        sample_params = rearrange(sample_params,'b c n 1 1 -> b n c')
        num_layers = len(self.channels) - 1
        for idx, nci in enumerate(self.channels[:-1]):
            nco = self.channels[idx + 1]

            bstart, bstop = self._get_bias_indices(idx)
            b_ = sample_params[:,:,bstart:bstop]
            out = self.net[idx](out)
            #print('shape', out.shape, b_.shape)
            out = out + b_
            # Apply RelU non-linearity in all but the last layer, and tanh in the last
            if idx < num_layers - 1:
                if self.use_sine:
                    out = self.sine(out)
                else:
                    out = torch.nn.functional.leaky_relu(out, 0.01, inplace=True)
        out = rearrange(out,'bs n 1 -> bs n', n=n)
        return out

class NFCoordSirenMultiDecoder3D(BaseNetwork):
    """pixel-wise MLP with 3D(x,y,z) input"""
    def __init__(self, num_outputs=1, latent_dim=128, dims=[64,64,64,64], maximal_period=96, last_tanh=False,use_sine=True):
        super(NFCoordSirenMultiDecoder3D, self).__init__()

        self.num_outputs = num_outputs
        self.last_tanh = last_tanh
        self.use_sine = use_sine
        self.num_params = latent_dim
        self.channels = [3 + self.num_params] + dims + [num_outputs]
        latent_drop = False
        self.biases = []
        print(self.channels)
        self.net = nn.ModuleList()
        self.sine = SirenActivation()
        for i in range(len(self.channels) - 1):
            lin = nn.Linear(self.channels[i], self.channels[i+1], bias=True)
            self.net.append(lin)
        if latent_drop:
            self.lat_drop = nn.Dropout(0.2)
        else:
            self.lat_drop = None
        
        if self.use_sine:
            print('now use siren')
            self.apply(self.sine.sine_init)
            self.net[0].apply(self.sine.first_layer_sine_init)

    def _set_num_params(self):
        nparams = 0
        # go over input/output channels for each layer
        idx = 0
        for layer, nci in enumerate(self.channels[:-1]):
            nco = self.channels[layer + 1] 
            nparams += nco  # FC biases
            self.biases.append((idx, idx + nco))
            idx += nco
        self.num_params = nparams

    def _get_bias_indices(self, idx):
        return self.biases[idx]

    def forward(self, coords, lr_params):
        bs, t, n, p = coords.shape
        out = coords
        #coords = coords.reshape(bs, n, 1, 1, p)
        
        out = rearrange(coords, 'bs t n p -> (bs t) n p')
        coords = rearrange(coords, 'bs t n p -> (bs t) n 1 1 p')
        lr_params = rearrange(lr_params, 'b t c d h w -> (b t) c d h w')
        
        sample_params = F.grid_sample(lr_params, coords, padding_mode='border', align_corners=True)
        sample_params = rearrange(sample_params,'b c n 1 1 -> b n c')
        out = torch.concat([out, sample_params], dim=2)
        num_layers = len(self.channels) - 1
        for idx, nci in enumerate(self.channels[:-1]):
            nco = self.channels[idx + 1]
            out = self.net[idx](out)
            # Apply RelU non-linearity in all but the last layer, and tanh in the last
            if idx < num_layers - 1:
                if self.use_sine:
                    out = self.sine(out)
                else:
                    out = torch.nn.functional.leaky_relu(out, 0.01, inplace=True)
            else:
                if self.last_tanh:
                    out = F.tanh(out)
                else:
                    out = out
        #out = rearrange(out,'bs n 1 -> bs n', n=n)
        out = rearrange(out,'(b t) n 1 -> b t n', b=bs,t=t)
        return out

class NFCoordSirenSimpleDecoder3D(BaseNetwork):
    """pixel-wise MLP with 3D(x,y,z) input"""
    def __init__(self, num_outputs=1, latent_dim=128, dims=[64,64,64,64], maximal_period=96, last_tanh=False, use_sine=True):
        super(NFCoordSirenSimpleDecoder3D, self).__init__()

        self.num_outputs = num_outputs
        self.last_tanh = last_tanh
        self.use_sine = use_sine
        self.num_params = latent_dim
        self.channels = [3 + self.num_params] + dims + [num_outputs]
        latent_drop = False
        self.biases = []
        print(self.channels)
        self.net = nn.ModuleList()
        self.sine = SirenActivation()
        for i in range(len(self.channels) - 1):
            lin = nn.Linear(self.channels[i], self.channels[i+1], bias=True)
            self.net.append(lin)
        if latent_drop:
            self.lat_drop = nn.Dropout(0.2)
        else:
            self.lat_drop = None
        
        if self.use_sine:
            print('now use siren')
            self.apply(self.sine.sine_init)
            self.net[0].apply(self.sine.first_layer_sine_init)

    def _set_num_params(self):
        nparams = 0
        # go over input/output channels for each layer
        idx = 0
        for layer, nci in enumerate(self.channels[:-1]):
            nco = self.channels[layer + 1] 
            nparams += nco  # FC biases
            self.biases.append((idx, idx + nco))
            idx += nco
        self.num_params = nparams

    def _get_bias_indices(self, idx):
        return self.biases[idx]

    def forward(self, coords, lr_params):
        bs, n, p = coords.shape
        out = coords
        #coords = coords.reshape(bs, n, 1, 1, p)
        #sample_params = F.grid_sample(lr_params, coords, padding_mode='border', align_corners=True)
        #sample_params = rearrange(sample_params,'b c n 1 1 -> b n c')
        sample_params = repeat(lr_params,'b 1 c -> b n c',n=n)
        out = torch.concat([out, sample_params], dim=2)
        num_layers = len(self.channels) - 1
        for idx, nci in enumerate(self.channels[:-1]):
            nco = self.channels[idx + 1]
            out = self.net[idx](out)
            # Apply RelU non-linearity in all but the last layer, and tanh in the last
            if idx < num_layers - 1:
                if self.use_sine:
                    out = self.sine(out)
                else:
                    out = torch.nn.functional.leaky_relu(out, 0.01, inplace=True)
            else:
                if self.last_tanh:
                    out = F.tanh(out)
                else:
                    out = out
        out = rearrange(out,'bs n 1 -> bs n', n=n)
        return out

class NFCoordSirenDecoder3D(BaseNetwork):
    """pixel-wise MLP with 3D(x,y,z) input"""
    def __init__(self, num_outputs=1, latent_dim=128, dims=[64,64,64,64], maximal_period=96, last_tanh=False, use_sine=True):
        super(NFCoordSirenDecoder3D, self).__init__()

        self.num_outputs = num_outputs
        self.last_tanh = last_tanh
        self.use_sine = use_sine
        self.num_params = latent_dim
        self.channels = [3 + self.num_params] + dims + [num_outputs]
        latent_drop = False
        self.biases = []
        print(self.channels)
        self.net = nn.ModuleList()
        self.sine = SirenActivation()
        for i in range(len(self.channels) - 1):
            lin = nn.Linear(self.channels[i], self.channels[i+1], bias=True)
            self.net.append(lin)
        if latent_drop:
            self.lat_drop = nn.Dropout(0.2)
        else:
            self.lat_drop = None
        
        if self.use_sine:
            print('now use siren')
            self.apply(self.sine.sine_init)
            self.net[0].apply(self.sine.first_layer_sine_init)

    def _set_num_params(self):
        nparams = 0
        # go over input/output channels for each layer
        idx = 0
        for layer, nci in enumerate(self.channels[:-1]):
            nco = self.channels[layer + 1] 
            nparams += nco  # FC biases
            self.biases.append((idx, idx + nco))
            idx += nco
        self.num_params = nparams

    def _get_bias_indices(self, idx):
        return self.biases[idx]

    def forward(self, coords, lr_params):
        bs, n, p = coords.shape
        out = coords
        coords = coords.reshape(bs, n, 1, 1, p)
        sample_params = F.grid_sample(lr_params, coords, padding_mode='border', align_corners=True)
        sample_params = rearrange(sample_params,'b c n 1 1 -> b n c')
        out = torch.concat([out, sample_params], dim=2)
        num_layers = len(self.channels) - 1
        for idx, nci in enumerate(self.channels[:-1]):
            nco = self.channels[idx + 1]
            out = self.net[idx](out)
            # Apply RelU non-linearity in all but the last layer, and tanh in the last
            if idx < num_layers - 1:
                if self.use_sine:
                    out = self.sine(out)
                else:
                    out = torch.nn.functional.leaky_relu(out, 0.01, inplace=True)
            else:
                if self.last_tanh:
                    out = F.tanh(out)
                else:
                    out = out
        out = rearrange(out,'bs n 1 -> bs n', n=n)
        return out

class NFCoordConcatDecoder3D(BaseNetwork):
    """pixel-wise MLP with 3D(x,y,z) input"""
    def __init__(self, num_outputs=1, latent_dim=128, dims=[64,64,64,64], maximal_period=96, use_sine=False):
        super(NFCoordConcatDecoder3D, self).__init__()

        self.num_outputs = num_outputs
        self.pe = positional_encoding()      
        self.use_sine = use_sine
        self.num_params = latent_dim
        self.channels = [3 + self.num_params] + dims + [num_outputs]
        latent_drop = False
        self.biases = []
        print(self.channels)
        self.net = nn.ModuleList()
        for i in range(len(self.channels) - 1):
            lin = nn.Linear(self.channels[i], self.channels[i+1], bias=True)
            if i == len(self.channels) - 2:
                torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi)/ np.sqrt(self.channels[i]), std=0.00001)
                torch.nn.init.constant_(lin.bias, -1)
            else:
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(self.channels[i+1]))
                torch.nn.init.constant_(lin.bias, 0.0)
            self.net.append(lin)
        if latent_drop:
            self.lat_drop = nn.Dropout(0.2)
        else:
            self.lat_drop = None
        self.init = InitWeights_me('xavier', 0.02)
        self.apply(self.init)

    def _set_num_params(self):
        nparams = 0
        # go over input/output channels for each layer
        idx = 0
        for layer, nci in enumerate(self.channels[:-1]):
            nco = self.channels[layer + 1] 
            nparams += nco  # FC biases
            self.biases.append((idx, idx + nco))
            idx += nco
        self.num_params = nparams

    def _get_bias_indices(self, idx):
        return self.biases[idx]

    def forward(self, coords, lr_params):
        bs, n, p = coords.shape
        #out = self.pe(coords) ## ppositional encoding
        out = coords
        #print(out.shape)
        coords = coords.reshape(bs, n, 1, 1, p)
        #sample_params = self.grid_sample(lr_params, coords)
        sample_params = F.grid_sample(lr_params, coords, padding_mode='border', align_corners=True)
        sample_params = rearrange(sample_params,'b c n 1 1 -> b n c')
        out = torch.concat([out, sample_params], dim=2)
        num_layers = len(self.channels) - 1
        for idx, nci in enumerate(self.channels[:-1]):
            nco = self.channels[idx + 1]
            out = self.net[idx](out)
            # Apply RelU non-linearity in all but the last layer, and tanh in the last
            if idx < num_layers - 1:
                out = torch.nn.functional.leaky_relu(out, 0.01, inplace=True)
            else:
                out = F.tanh(out)
        out = rearrange(out,'bs n 1 -> bs n', n=n)
        return out

class NFCoordConcatDecoder4D(BaseNetwork):
    """pixel-wise MLP with 3D(x,y,z) input"""
    def __init__(self, num_outputs=1, latent_dim=128, dims=[64,64,64,64], maximal_period=96, use_sine=False):
        super(NFCoordConcatDecoder4D, self).__init__()

        self.num_outputs = num_outputs
        self.use_sine = use_sine
        self.num_params = latent_dim
        self.channels = [4 + self.num_params] + dims + [num_outputs]
        latent_drop = False
        self.biases = []
        print(self.channels)
        self.net = nn.ModuleList()
        for i in range(len(self.channels) - 1):
            lin = nn.Linear(self.channels[i], self.channels[i+1], bias=True)
            if i == len(self.channels) - 2:
                torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi)/ np.sqrt(self.channels[i]), std=0.00001)
                torch.nn.init.constant_(lin.bias, -1)
            else:
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(self.channels[i+1]))
                torch.nn.init.constant_(lin.bias, 0.0)
            self.net.append(lin)
        if latent_drop:
            self.lat_drop = nn.Dropout(0.2)
        else:
            self.lat_drop = None

    def grid_sample(self, image, optical):
        N, C, ID, IH, IW = image.shape
        _, D, H, W, _ = optical.shape

        ix = optical[..., 0]
        iy = optical[..., 1]
        iz = optical[..., 2]

        ix = ((ix + 1) / 2) * (IW - 1);
        iy = ((iy + 1) / 2) * (IH - 1);
        iz = ((iz + 1) / 2) * (ID - 1);
        with torch.no_grad():
        
            ix_tnw = torch.floor(ix);
            iy_tnw = torch.floor(iy);
            iz_tnw = torch.floor(iz);

            ix_tne = ix_tnw + 1;
            iy_tne = iy_tnw;
            iz_tne = iz_tnw;

            ix_tsw = ix_tnw;
            iy_tsw = iy_tnw + 1;
            iz_tsw = iz_tnw;

            ix_tse = ix_tnw + 1;
            iy_tse = iy_tnw + 1;
            iz_tse = iz_tnw;

            ix_bnw = ix_tnw;
            iy_bnw = iy_tnw;
            iz_bnw = iz_tnw + 1;

            ix_bne = ix_tnw + 1;
            iy_bne = iy_tnw;
            iz_bne = iz_tnw + 1;

            ix_bsw = ix_tnw;
            iy_bsw = iy_tnw + 1;
            iz_bsw = iz_tnw + 1;

            ix_bse = ix_tnw + 1;
            iy_bse = iy_tnw + 1;
            iz_bse = iz_tnw + 1;

        tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
        tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
        tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
        tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
        bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
        bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
        bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
        bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);


        with torch.no_grad():

            torch.clamp(ix_tnw, 0, IW - 1, out=ix_tnw)
            torch.clamp(iy_tnw, 0, IH - 1, out=iy_tnw)
            torch.clamp(iz_tnw, 0, ID - 1, out=iz_tnw)

            torch.clamp(ix_tne, 0, IW - 1, out=ix_tne)
            torch.clamp(iy_tne, 0, IH - 1, out=iy_tne)
            torch.clamp(iz_tne, 0, ID - 1, out=iz_tne)

            torch.clamp(ix_tsw, 0, IW - 1, out=ix_tsw)
            torch.clamp(iy_tsw, 0, IH - 1, out=iy_tsw)
            torch.clamp(iz_tsw, 0, ID - 1, out=iz_tsw)

            torch.clamp(ix_tse, 0, IW - 1, out=ix_tse)
            torch.clamp(iy_tse, 0, IH - 1, out=iy_tse)
            torch.clamp(iz_tse, 0, ID - 1, out=iz_tse)

            torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
            torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
            torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

            torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
            torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
            torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)

            torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
            torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
            torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

            torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
            torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
            torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

        image = image.view(N, C, ID * IH * IW)

        tnw_val = torch.gather(image, 2, (iz_tnw * IW * IH + iy_tnw * IW + ix_tnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
        tne_val = torch.gather(image, 2, (iz_tne * IW * IH + iy_tne * IW + ix_tne).long().view(N, 1, D * H * W).repeat(1, C, 1))
        tsw_val = torch.gather(image, 2, (iz_tsw * IW * IH + iy_tsw * IW + ix_tsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
        tse_val = torch.gather(image, 2, (iz_tse * IW * IH + iy_tse * IW + ix_tse).long().view(N, 1, D * H * W).repeat(1, C, 1))
        bnw_val = torch.gather(image, 2, (iz_bnw * IW * IH + iy_bnw * IW + ix_bnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
        bne_val = torch.gather(image, 2, (iz_bne * IW * IH + iy_bne * IW + ix_bne).long().view(N, 1, D * H * W).repeat(1, C, 1))
        bsw_val = torch.gather(image, 2, (iz_bsw * IW * IH + iy_bsw * IW + ix_bsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
        bse_val = torch.gather(image, 2, (iz_bse * IW * IH + iy_bse * IW + ix_bse).long().view(N, 1, D * H * W).repeat(1, C, 1))

        out_val = (tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W) +
               tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W) +
               tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W) +
               tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W) +
               bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
               bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
               bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
               bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W))

        return out_val
    def _set_num_params(self):
        nparams = 0
        # go over input/output channels for each layer
        idx = 0
        for layer, nci in enumerate(self.channels[:-1]):
            nco = self.channels[layer + 1] 
            nparams += nco  # FC biases
            self.biases.append((idx, idx + nco))
            idx += nco
        self.num_params = nparams

    def _get_bias_indices(self, idx):
        return self.biases[idx]

    def forward(self, coords, lr_params, study_dates):
        #print(coords.shape, lr_params.shape, study_dates.shape)
        bs, t, n, p = coords.shape
        coords = rearrange(coords, 'bs t n p -> bs (t n) p')
        #bs, n, p = coords.shape
        out = coords
        coords = coords.reshape(bs, t * n, 1, 1, p)
        #sample_params = self.grid_sample(lr_params, coords)
        sample_params = F.grid_sample(lr_params, coords, padding_mode='border', align_corners=True)
        sample_params = rearrange(sample_params,'b c n 1 1 -> b n c')
        #print(sample_params.shape)
        study_dates = study_dates.repeat_interleave(n,dim=1).unsqueeze(2)
        #print(study_dates.shape)
        #print(study_dates[0,0,0],study_dates[0,1,0], study_dates[0,2,0])
        out = torch.concat([out, study_dates, sample_params], dim=2)
        num_layers = len(self.channels) - 1
        for idx, nci in enumerate(self.channels[:-1]):
            nco = self.channels[idx + 1]
            out = self.net[idx](out)
            # Apply RelU non-linearity in all but the last layer, and tanh in the last
            if idx < num_layers - 1:
                out = torch.nn.functional.leaky_relu(out, 0.01, inplace=True)
            else:
                out = F.tanh(out)
        #print('out', out.shape)
        out = rearrange(out,'bs (t n) 1 -> bs t n',t=t,n=n)
        #out = out.squeeze(2)
        return out

class ImplicitNet(BaseNetwork):
    def __init__(
        self,
        d_in,
        dims,
        skip_in=(),
        geometric_init=True,
        radius_init=1,
        beta=100
    ):
        super().__init__()

        dims = [d_in] + dims + [1]

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for layer in range(0, self.num_layers - 1):

            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim)

            # if true preform preform geometric initialization
            if geometric_init:

                if layer == self.num_layers - 2:

                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)

                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            setattr(self, "lin" + str(layer), lin)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)

        # vanilla relu
        else:
            self.activation = nn.ReLU()

    def forward(self, coords, params):

        bs, n, p = coords.shape
        #x = coords
        coords = coords.reshape(bs, n, 1, 1, p)
        sample_params = F.grid_sample(params, coords)
        #sample_params = self.grid_sample(lr_params, coords)
        sample_params = rearrange(sample_params,'b c n 1 1 -> b n c')
        coords = coords.reshape(bs, n, p)
        input = torch.concat([coords, sample_params], dim=2)
        x = input

        for layer in range(0, self.num_layers - 1):

            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = lin(x)

            if layer < self.num_layers - 2:
                x = self.activation(x)

        x = F.tanh(x)
        x = x.squeeze(2)
        return x
