from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
#from models.networks.generator_cat import Encoder_fpn4
from models.networks.encoder import ConesEncoder, Conv3dEncoder, Conv3dEncoder_simple
from models.networks.decoder import NF3dDecoder, NF4dWoiDecoder, NF4dDecoder, NF4dDecoder_simple
import torch as th
from math import pi
from models.networks.generator import InitWeights_me, Bottleneck3d

class LongMri4DGenerator(BaseNetwork):
    def __init__(self, opt, num_input):
        super(LongMri4DGenerator, self).__init__()
        self.num_inputs = num_input
        self.gpu_ids = opt.gpu_ids
        self.max_period = opt.maximal_period
        self.decoder = NF4dDecoder(num_inputs=2, num_outputs=1, maximal_period=self.max_period, width=opt.hr_width,
                                    depth=opt.hr_depth, coordinates=opt.hr_coor)

        num_params = self.decoder.num_params
        
        self.latlayers = nn.Conv3d(256, num_params, kernel_size=1, stride=1,padding=0)
        self.encoder = Conv3dEncoder_simple(num_in=self.num_inputs, block=Bottleneck3d, num_blocks=[2,4,23,3])
        self.bn = nn.InstanceNorm3d(256, affine=True)
        self.init = InitWeights_me(opt.init_type, opt.init_variance)
        self.encoder.apply(self.init)
        self.latlayers.apply(self.init)
        self.decoder.apply(self.init)
    
    def use_gpu(self):
        return len(self.gpu_ids) > 0
    
    def forward(self, input, study_dates):
        features = self.encoder(input)
        features = self.bn(features)
        features = self.latlayers(features)
        output = self.decoder(input, study_dates, features)
        return output, features

