from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.decoder import NFCoordConcatDecoder3D
from models.networks.encoder import PointEncoder
import torch
from models.networks.sdfgenerator import DeepSDF

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
        self.bn1 = nn.InstanceNorm3d(planes, affine=True)
        #self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm3d(planes, affine=True)
        #self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.InstanceNorm3d(self.expansion*planes, affine=True)
        #self.bn3 = nn.BatchNorm3d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm3d(self.expansion*planes, affine=True)
                #nn.BatchNorm3d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class LongMriCoord3DDeepGenerator(BaseNetwork):
    def __init__(self, opt, num_input, use_sine):
        super(LongMriCoord3DDeepGenerator, self).__init__()
        self.num_inputs = num_input
        self.gpu_ids = opt.gpu_ids
        self.use_sine = use_sine
        self.max_period = opt.maximal_period
        self.init = InitWeights_me(opt.init_type, opt.init_variance)
        #self.dims = [512, 512, 512, 512, 512, 512, 512,512]
        self.dims = [512, 512, 512, 512, 512, 512, 512,512]
        self.dropout = [0,1,2,3,4,5,6,7]
        self.norm_layer = [0,1,2,3,4,5,6,7]
        num_params = 256

        self.decoder = NFCoordConcatDecoder3D(num_outputs=1, latent_dim=num_params)
        #self.decoder = DeepSDF(latent_size=256, dims=self.dims, dropout=self.dropout, dropout_prob=0.2, norm_layers=self.norm_layer)       
        #self.encoder = ConvDown3DEncoder(num_in=self.num_inputs, num_out=256, block=Bottleneck3d, num_blocks=[2,2,2,2])
        self.encoder = EncoderSimple3D(num_in=self.num_inputs, num_out=256)
        
        self.bn = nn.InstanceNorm3d(256, affine=True)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.latlayers = nn.Linear(256, num_params)
        self.encoder.apply(self.init)
        self.latlayers.apply(self.init)
        self.decoder.apply(self.init)
    
    def forward(self, input, coords):
        features = self.encoder(input)
        features = self.pool(features)
        bs, c, _, _, _ = features.shape
        features = features.reshape(bs, c)
        features = self.latlayers(features)
        output = self.decoder(coords, features)
        return output, features


class LongMriCoord3DPatchConcatGenerator(BaseNetwork):
    def __init__(self, opt, num_input, use_sine):
        super(LongMriCoord3DPatchConcatGenerator, self).__init__()
        self.num_inputs = num_input
        self.gpu_ids = opt.gpu_ids
        self.use_sine = use_sine
        self.max_period = opt.maximal_period
        self.init = InitWeights_me(opt.init_type, opt.init_variance)
        self.dims = [64,64,64,64]
        self.decoder = NFCoordPatchConcatDecoder3D(num_outputs=1, dims=self.dims, maximal_period=self.max_period, use_sine=self.use_sine)
        self.latent_dim = 64
        self.encoder = ConvDown3DEncoder(num_in=self.num_inputs, num_out=256, block=Bottleneck3d, num_blocks=[2,2,2,2])
        self.bn = nn.InstanceNorm3d(256, affine=True)
        self.latlayers = nn.Conv3d(256, self.latent_dim, kernel_size=1, stride=1,padding=0)
        self.encoder.apply(self.init)
        self.latlayers.apply(self.init)
        if not self.use_sine:
            self.decoder.apply(self.init)
    
    def use_gpu(self):
        return len(self.gpu_ids) > 0
    
    def forward(self, input, coords):
        features = self.encoder(input)
        features = F.relu(self.bn(features))
        features = self.latlayers(features)
        output = self.decoder(coords, features)
        return output, features

class LongMriCoord3DPatchGenerator(BaseNetwork):
    def __init__(self, opt, num_input, use_sine):
        super(LongMriCoord3DPatchGenerator, self).__init__()
        self.num_inputs = num_input
        self.gpu_ids = opt.gpu_ids
        self.use_sine = use_sine
        self.max_period = opt.maximal_period
        self.init = InitWeights_me(opt.init_type, opt.init_variance)
        self.dims = [64,64,64,64]
        self.decoder = NFCoordPatchDecoder3D(num_outputs=1, dims=self.dims, maximal_period=self.max_period, use_sine=self.use_sine)
        num_params = self.decoder.num_params
        self.encoder = ConvDown3DEncoder(num_in=self.num_inputs, num_out=256, block=Bottleneck3d, num_blocks=[2,2,2,2])
        self.bn = nn.InstanceNorm3d(256, affine=True)
        self.latlayers = nn.Conv3d(256, num_params, kernel_size=1, stride=1,padding=0)
        self.encoder.apply(self.init)
        self.latlayers.apply(self.init)
        if not self.use_sine:
            self.decoder.apply(self.init)
    
    def use_gpu(self):
        return len(self.gpu_ids) > 0
    
    def forward(self, input, coords):
        features = self.encoder(input)
        features = F.relu(self.bn(features))
        features = self.latlayers(features)
        print(features.shape)
        output = self.decoder(coords, features)
        return output, features

class LongMripntGenerator(BaseNetwork):
    def __init__(self, opt, num_input, use_sine):
        super(LongMripntGenerator, self).__init__()
        self.num_inputs = num_input
        self.gpu_ids = opt.gpu_ids
        self.use_sine = use_sine
        self.max_period = opt.maximal_period
        self.init = InitWeights_me(opt.init_type, opt.init_variance)
        self.dims = [64,64,64,64]
        self.decoder = NFCoordfullDecoder3D(num_outputs=1, dims=self.dims, maximal_period=self.max_period, use_sine=self.use_sine)
        self.num_params = self.decoder.num_params
        self.encoder = PointEncoder(num_in=3, dim=32, hidden_dim=32, n_blocks=5)
        #self.bn = nn.InstanceNorm3d(256, affine=True)
        #self.bn = nn.BatchNorm3d(256)
        #self.latlayers = nn.Conv3d(256, self.num_params, kernel_size=1, stride=1,padding=0)
        self.encoder.apply(self.init)
        #self.latlayers.apply(self.init)
        if not self.use_sine:
            self.decoder.apply(self.init)
    
    def forward(self, coords):
        features = self.encoder(coords)
        #features = self.bn(features)
        #features = self.latlayers(features)
        print('features', features.shape)
        output = self.decoder(coords, features)
        return output, features

class LongMriCoord3DfullFMGenerator(BaseNetwork):
    def __init__(self, opt, num_input, use_sine):
        super(LongMriCoord3DfullFMGenerator, self).__init__()
        self.num_inputs = num_input
        self.gpu_ids = opt.gpu_ids
        self.use_sine = use_sine
        self.max_period = opt.maximal_period
        self.init = InitWeights_me(opt.init_type, opt.init_variance)
        self.dims = [64,64,64,64]
        self.decoder = NFCoordfullDecoder3D(num_outputs=1, dims=self.dims, maximal_period=self.max_period, use_sine=self.use_sine)
        self.num_params = self.decoder.num_params
        self.encoder = Conv3dEncoder(num_in=self.num_inputs, block=Bottleneck3d, num_blocks=[2,2,2,2])
        self.bn = nn.InstanceNorm3d(256, affine=True)
        #self.bn = nn.BatchNorm3d(256)
        self.latlayers = nn.Conv3d(256, self.num_params, kernel_size=1, stride=1,padding=0)

        self.encoder.apply(self.init)
        self.latlayers.apply(self.init)
        self.decoder.apply(self.init)
        #if not self.use_sine:
        #    self.decoder.apply(self.init)
    
    def forward(self, input, coords):
        features = self.encoder(input)
        features = self.bn(features)
        features = self.latlayers(features)
        output = self.decoder(coords, features)
        return output, features

class LongMriCoordConcat3DIGRGenerator(BaseNetwork):
    def __init__(self, opt, num_input, use_sine):
        super(LongMriCoordConcat3DIGRGenerator, self).__init__()
        self.num_inputs = num_input
        self.gpu_ids = opt.gpu_ids
        self.use_sine = use_sine
        self.max_period = opt.maximal_period
        self.init = InitWeights_me(opt.init_type, opt.init_variance)
        #self.dims = [64,64,64,64]
        self.dims = [512,512,512,512,512,512,512,512]
        self.skip_in = [4]
        self.num_params = 128
        #self.decoder = NFCoordConcatDecoder3D(num_outputs=1, latent_dim=self.num_params, dims=self.dims, maximal_period=self.max_period, use_sine=self.use_sine)
        self.decoder = ImplicitNet(d_in=(3+self.num_params), dims=self.dims, skip_in=self.skip_in)
        self.encoder = ConvDown3DEncoder(num_in=self.num_inputs, num_out=256, block=Bottleneck3d, num_blocks=[2,2,2,2])
        
        self.bn = nn.InstanceNorm3d(256, affine=True)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.latlayers = nn.Linear(256, self.num_params)
        self.encoder.apply(self.init)
        self.latlayers.apply(self.init)
        #if not self.use_sine:
        #    self.decoder.apply(self.init)
    
    def use_gpu(self):
        return len(self.gpu_ids) > 0
    
    def forward(self, input, coords):
        features = self.encoder(input)
        #features = F.relu(self.bn(features))
        #features = self.bn(features)
        features = self.pool(features)
        bs, c, _, _, _ = features.shape
        features = features.reshape(bs, c)
        features = self.latlayers(features)
        _,n,_ = coords.shape
        features_flat = features.unsqueeze(1).expand(-1, n, -1)
        input = torch.concat([features_flat, coords], dim=2)
        output = self.decoder(input)
        return output, features

class LongMriCoordConcat3DGenerator(BaseNetwork):
    def __init__(self, opt, num_input, use_sine):
        super(LongMriCoordConcat3DGenerator, self).__init__()
        self.num_inputs = num_input
        self.gpu_ids = opt.gpu_ids
        self.use_sine = use_sine
        self.max_period = opt.maximal_period
        self.init = InitWeights_me(opt.init_type, opt.init_variance)
        self.dims = [64,64,64,64]
        self.num_params = 128
        self.decoder = NFCoordConcatDecoder3D(num_outputs=1, latent_dim=self.num_params, dims=self.dims, maximal_period=self.max_period, use_sine=self.use_sine)
        self.encoder = ConvDown3DEncoder(num_in=self.num_inputs, num_out=256, block=Bottleneck3d, num_blocks=[2,2,2,2])
        
        self.bn = nn.InstanceNorm3d(256, affine=True)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.latlayers = nn.Linear(256, self.num_params)
        self.encoder.apply(self.init)
        self.latlayers.apply(self.init)
        if not self.use_sine:
            self.decoder.apply(self.init)
    
    def use_gpu(self):
        return len(self.gpu_ids) > 0
    
    def forward(self, input, coords):
        features = self.encoder(input)
        #features = F.relu(self.bn(features))
        #features = self.bn(features)
        features = self.pool(features)
        bs, c, _, _, _ = features.shape
        features = features.reshape(bs, c)
        features = self.latlayers(features)
        output = self.decoder(coords, features)
        return output, features

class LongMriCoord3DGenerator(BaseNetwork):
    def __init__(self, opt, num_input, use_sine):
        super(LongMriCoord3DGenerator, self).__init__()
        self.num_inputs = num_input
        self.gpu_ids = opt.gpu_ids
        self.use_sine = use_sine
        self.max_period = opt.maximal_period
        self.init = InitWeights_me(opt.init_type, opt.init_variance)
        self.dims = [64,64,64,64]
        self.decoder = NFCoordDecoder3D(num_outputs=1, dims=self.dims, maximal_period=self.max_period, use_sine=self.use_sine)
        num_params = self.decoder.num_params
        self.encoder = ConvDown3DEncoder(num_in=self.num_inputs, num_out=256, block=Bottleneck3d, num_blocks=[2,2,2,2])
        
        self.bn = nn.InstanceNorm3d(256, affine=True)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.latlayers = nn.Linear(256, num_params)
        self.encoder.apply(self.init)
        self.latlayers.apply(self.init)
        if not self.use_sine:
            self.decoder.apply(self.init)
    
    def use_gpu(self):
        return len(self.gpu_ids) > 0
    
    def forward(self, input, coords):
        features = self.encoder(input)
        features = F.relu(self.bn(features))
        features = self.pool(features)
        bs, c, _, _, _ = features.shape
        features = features.reshape(bs, c)
        features = self.latlayers(features)
        output = self.decoder(coords, features)
        return output, features

class LongMriCoord4DGenerator(BaseNetwork):
    def __init__(self, opt, num_input, use_sine):
        super(LongMriCoord4DGenerator, self).__init__()
        self.num_inputs = num_input
        self.gpu_ids = opt.gpu_ids
        self.use_sine = use_sine
        self.max_period = opt.maximal_period
        self.init = InitWeights_me(opt.init_type, opt.init_variance)
        #self.dims = [128,128,128,192,128,128,192,128,128,128]
        self.dims = [64,64,64,64]
        self.decoder = NFCoordDecoder4D(num_outputs=1, dims=self.dims, maximal_period=self.max_period, use_sine=self.use_sine)

        num_params = self.decoder.num_params
        
        #self.encoder = EncoderSimple3D(num_in=self.num_inputs, num_out=1024)
        self.encoder = ConvDown3DEncoder(num_in=self.num_inputs, num_out=256, block=Bottleneck3d, num_blocks=[2,2,2,2])
        
        self.bn = nn.InstanceNorm3d(256, affine=True)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.latlayers = nn.Linear(256, num_params)
        self.encoder.apply(self.init)
        self.latlayers.apply(self.init)
        if not self.use_sine:
            self.decoder.apply(self.init)
    
    def use_gpu(self):
        return len(self.gpu_ids) > 0
    
    def forward(self, input, coords):
        features = self.encoder(input)
        features = F.relu(self.bn(features))
        features = self.pool(features)
        bs, c, _, _, _ = features.shape
        #features = features.squeeze(2).squeeze(2).squeeze(2)
        features = features.reshape(bs, c)
        features = self.latlayers(features)
        output = self.decoder(input, coords, features)
        return output, features

