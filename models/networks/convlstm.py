from re import A
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import torch
from models.networks.base_network import BaseNetwork
from util.util import InitWeights_me, grid_sample

class STConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(STConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + 2 * self.hidden_dim,
                              out_channels=5 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state, prev_state):
        h_cur, c_cur = cur_state
        h_prev,  c_prev = prev_state
        combined = torch.cat([input_tensor, h_cur, h_prev], dim=1)  # concatenate along channel axis
        
        combined_conv = self.conv(combined)
        cc_i, cc_fs, cc_ft, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        fs = torch.sigmoid(cc_fs)
        ft = torch.sigmoid(cc_ft)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = fs * c_cur +  ft * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class Conv3DLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(Conv3DLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2
        self.bias = bias

        self.conv = nn.Conv3d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        depth, height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, depth, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, depth, height, width, device=self.conv.weight.device))

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class INRConv3DLSTM(BaseNetwork):
    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(INRConv3DLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(Conv3DLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
        #self.outimg_conv = nn.Conv3d(in_channels=hidden_dim[-1],out_channels=1,kernel_size=1)
        #self.outconv = nn.Conv3d(in_channels=hidden_dim[-1], out_channels=32,kernel_size=1)
        self.cell_list = nn.ModuleList(cell_list)
        self.init = InitWeights_me('xavier', 0.02)
        self.apply(self.init)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, d, h, w) -> (b, t, c, d, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4,5)

        b, s, _,d, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(d, h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :,:],
                                                 cur_state=[h, c])
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        #if not self.return_all_layers:
        #    layer_output_list = layer_output_list[-1:]
        #    last_state_list = last_state_list[-1:]
        
        layer_output = layer_output_list[-1]
        last_state = last_state_list[-1]
        #print(layer_output.shape)
        #layer_output = rearrange(layer_output,'b t c d h w -> (b t) c d h w')
        #layer_output = self.outconv(layer_output)
        #layer_output = rearrange(layer_output,'(b t) c d h w -> b t c d h w', b=b,t=s)
        return layer_output

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class Conv3DLSTM(BaseNetwork):
    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(Conv3DLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(Conv3DLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
        self.outimg_conv = nn.Conv3d(in_channels=hidden_dim[-1],out_channels=1,kernel_size=1)
        self.outseg_conv = nn.Conv3d(in_channels=hidden_dim[-1], out_channels=1,kernel_size=1)
        self.cell_list = nn.ModuleList(cell_list)
        self.init = InitWeights_me('xavier', 0.02)
        self.apply(self.init)

    def forward(self, input_tensor, study_dates, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, d, h, w) -> (b, t, c, d, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4,5)

        b, s, _,d, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(d, h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)
            #print('output', len(output_inner))
            layer_output = torch.stack(output_inner, dim=1)
            #print('output2', layer_output.shape)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        
        layer_output = layer_output_list[-1]
        last_state = last_state_list[-1]
        layer_output = rearrange(layer_output,'b t c d h w -> (b t) c d h w')
        output_img = self.outimg_conv(layer_output)
        output_seg = self.outseg_conv(layer_output)
        output_img = F.tanh(output_img)
        output_img = rearrange(output_img,'(b t) c d h w -> b t c d h w', b=b,t=s)
        output_seg = rearrange(output_seg,'(b t) c d h w -> b t c d h w', b=b,t=s)
        return output_img, output_seg, last_state

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class STConv3DLSTMSmall(BaseNetwork):
    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(STConv3DLSTMSmall, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        self.num_slice = 80
        self.num_sequence = 2

        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        
        '''
        cell_list_1 = []
        cell_list_2 = []
        for i in range(0, self.num_slice):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list_1.append(STConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
        for i in range(0,self.num_slice):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list_1.append(STConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
        '''
        self.lstmcell = STConvLSTMCell(input_dim=5,
                                          hidden_dim=8,
                                          kernel_size=(3,3),
                                          bias=self.bias)
        ### encoder
        self.conv1 = nn.Conv2d(in_channels=2,out_channels=8,kernel_size=3,stride=2,padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(in_channels=8,out_channels=8,kernel_size=3,stride=2,padding=1)
        self.bn3 = nn.BatchNorm2d(8)
        self.conv4 = nn.Conv2d(in_channels=8,out_channels=4,kernel_size=1,stride=1,padding=0)

        ## decoder
        self.deconv5 = nn.Conv2d(in_channels=8, out_channels=8,kernel_size=1,stride=1,padding=0)
        self.bn4 = nn.BatchNorm2d(8)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deconv6 = nn.Conv2d(in_channels=8, out_channels=8,kernel_size=3,stride=1,padding=1)
        self.bn5 = nn.BatchNorm2d(8)
        
        self.deconv7 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,stride=1,padding=1)
        self.bn6 = nn.BatchNorm2d(8)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deconv8 = nn.Conv2d(in_channels=8, out_channels=2 ,kernel_size=3,stride=1,padding=1)
        
        #self.outimg_conv = nn.Conv3d(in_channels=2, out_channels=1,kernel_size=1)
        #self.outseg_conv = nn.Conv3d(in_channels=2, out_channels=1,kernel_size=1)

        self.init = InitWeights_me('xavier', 0.02)
        self.apply(self.init)

    def forward(self, input_tensor, time_intervals, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, d, h, w) -> (b, t, c, d, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4,5)
        b, s, _, d, h, w = input_tensor.size()
        num_slice = d 
        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            ## Since the init is done in forward. Can send image size here
            #hidden_state = self._init_hidden(batch_size=b,
            #                                 image_size=(d, h, w))
            hidden_state = self.lstmcell.init_hidden(batch_size=b,image_size=(20,20))
        output_prev = []
        hidden_prev = []
        h, c = hidden_state
        for d_idx in range(num_slice):
            prev_hidden_state = self.lstmcell.init_hidden(batch_size=b,image_size=(20, 20))
            h_prev, c_prev = prev_hidden_state
            out = input_tensor[:, 0, :, d_idx,:]
            #encoder
            out = self.conv1(out)
            out = F.relu(self.bn1(out))
            out = self.conv2(out)
            out = F.relu(self.bn2(out))
            out = self.conv3(out)
            out = F.relu(self.bn3(out))
            out = self.conv4(out)
            
            ##lstmcell 
            time_factor = time_intervals[:,0]
            time_factor = rearrange(time_factor,'bs -> bs 1 1 1')
            time_factor = repeat(time_factor, 'bs 1 1 1 -> bs 1 20 20')
            out = torch.concat([out, time_factor], dim=1)
            h, c = self.lstmcell(input_tensor=out, cur_state=[h,c], prev_state=[h_prev,c_prev])
            hidden_prev.append((h,c))
            ## decoder
            out = self.deconv5(h)
            out = F.relu(self.bn4(out))
            out = self.upsample1(out)
            out = self.deconv6(out)
            out = F.relu(self.bn5(out))
            out = self.deconv7(out)
            out = F.relu(self.bn6(out))
            out = self.upsample2(out)
            out = self.deconv8(out)
            output_prev.append(out.unsqueeze(2))
        output_prev = torch.concat(output_prev,dim=2)
        
        output_pred = []
        h, c = hidden_prev[-1] ### last layer of previous scan
        for d_idx in range(num_slice):
            h_prev, c_prev = hidden_prev[d_idx]
            out = input_tensor[:,1,:, d_idx,:] ### 2nd time points
            ## encoder
            out = self.conv1(out)
            out = F.relu(self.bn1(out))
            out = self.conv2(out)
            out = F.relu(self.bn2(out))
            out = self.conv3(out)
            out = F.relu(self.bn3(out))
            out = self.conv4(out)
            ## lstmcell
            time_factor = time_intervals[:,1]
            time_factor = rearrange(time_factor,'bs -> bs 1 1 1')
            time_factor = repeat(time_factor, 'bs 1 1 1 -> bs 1 20 20')
            out = torch.concat([out, time_factor], dim=1)
            h, c = self.lstmcell(input_tensor=out, cur_state=[h,c], prev_state=[h_prev,c_prev])
            ## decoder
            out = self.deconv5(h)
            out = F.relu(self.bn4(out))
            out = self.upsample1(out)
            out = self.deconv6(out)
            out = F.relu(self.bn5(out))
            out = self.deconv7(out)
            out = F.relu(self.bn6(out))
            out = self.upsample2(out)
            out = self.deconv8(out)
            output_pred.append(out.unsqueeze(2))
        
        output_pred = torch.concat(output_pred,dim=2)
        final_output = torch.concat([output_prev.unsqueeze(1), output_pred.unsqueeze(1)],dim=1)
        
        output_img = final_output[:,:,0,:]
        #output_img = F.tanh(output_img)
        output_seg = final_output[:,:,1,:].unsqueeze(2)
        #print(output_img.shape, output_seg.shape)
        return output_img, output_seg, None

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
class STConv3DLSTM(BaseNetwork):
    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(STConv3DLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        #self.num_slice = 80
        self.num_sequence = 2

        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        
        '''
        cell_list_1 = []
        cell_list_2 = []
        for i in range(0, self.num_slice):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list_1.append(STConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
        for i in range(0,self.num_slice):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list_1.append(STConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
        '''
        self.lstmcell = STConvLSTMCell(input_dim=65,
                                          hidden_dim=64,
                                          kernel_size=(3,3),
                                          bias=self.bias)
        ### encoder
        self.conv1 = nn.Conv2d(in_channels=2,out_channels=8,kernel_size=3,stride=2,padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=2,padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=1,stride=1,padding=0)

        ## decoder
        self.deconv5 = nn.Conv2d(in_channels=64, out_channels=64,kernel_size=1,stride=1,padding=0)
        self.bn4 = nn.BatchNorm2d(64)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deconv6 = nn.Conv2d(in_channels=64, out_channels=32,kernel_size=3,stride=1,padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        
        self.deconv7 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3,stride=1,padding=1)
        self.bn6 = nn.BatchNorm2d(16)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deconv8 = nn.Conv2d(in_channels=16, out_channels=2 ,kernel_size=3,stride=1,padding=1)
        
        #self.outimg_conv = nn.Conv3d(in_channels=2, out_channels=1,kernel_size=1)
        #self.outseg_conv = nn.Conv3d(in_channels=2, out_channels=1,kernel_size=1)

        self.init = InitWeights_me('xavier', 0.02)
        self.apply(self.init)

    def forward(self, input_tensor, time_intervals, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, d, h, w) -> (b, t, c, d, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4,5)
        b, s, _, d, h, w = input_tensor.size()
        num_slice = d 
        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            ## Since the init is done in forward. Can send image size here
            #hidden_state = self._init_hidden(batch_size=b,
            #                                 image_size=(d, h, w))
            hidden_state = self.lstmcell.init_hidden(batch_size=b,image_size=(16,16))
        output_prev = []
        hidden_prev = []
        h, c = hidden_state
        #print('num', num_slice)
        for d_idx in range(num_slice):
            prev_hidden_state = self.lstmcell.init_hidden(batch_size=b,image_size=(16, 16))
            h_prev, c_prev = prev_hidden_state
            out = input_tensor[:, 0, :, d_idx,:]
            #encoder
            out = self.conv1(out)
            out = F.relu(self.bn1(out))
            out = self.conv2(out)
            out = F.relu(self.bn2(out))
            out = self.conv3(out)
            out = F.relu(self.bn3(out))
            out = self.conv4(out)
            
            ##lstmcell 
            time_factor = time_intervals[:,0]
            time_factor = rearrange(time_factor,'bs -> bs 1 1 1')
            time_factor = repeat(time_factor, 'bs 1 1 1 -> bs 1 16 16')
            out = torch.concat([out, time_factor], dim=1)
            h, c = self.lstmcell(input_tensor=out, cur_state=[h,c], prev_state=[h_prev,c_prev])
            hidden_prev.append((h,c))
            ## decoder
            out = self.deconv5(h)
            out = F.relu(self.bn4(out))
            out = self.upsample1(out)
            out = self.deconv6(out)
            out = F.relu(self.bn5(out))
            out = self.deconv7(out)
            out = F.relu(self.bn6(out))
            out = self.upsample2(out)
            out = self.deconv8(out)
            output_prev.append(out.unsqueeze(2))
        output_prev = torch.concat(output_prev,dim=2)
        
        output_pred = []
        h, c = hidden_prev[-1] ### last layer of previous scan
        for d_idx in range(num_slice):
            h_prev, c_prev = hidden_prev[d_idx]
            out = input_tensor[:,1,:, d_idx,:] ### 2nd time points
            ## encoder
            out = self.conv1(out)
            out = F.relu(self.bn1(out))
            out = self.conv2(out)
            out = F.relu(self.bn2(out))
            out = self.conv3(out)
            out = F.relu(self.bn3(out))
            out = self.conv4(out)
            ## lstmcell
            time_factor = time_intervals[:,1]
            time_factor = rearrange(time_factor,'bs -> bs 1 1 1')
            time_factor = repeat(time_factor, 'bs 1 1 1 -> bs 1 16 16')
            out = torch.concat([out, time_factor], dim=1)
            h, c = self.lstmcell(input_tensor=out, cur_state=[h,c], prev_state=[h_prev,c_prev])
            ## decoder
            out = self.deconv5(h)
            out = F.relu(self.bn4(out))
            out = self.upsample1(out)
            out = self.deconv6(out)
            out = F.relu(self.bn5(out))
            out = self.deconv7(out)
            out = F.relu(self.bn6(out))
            out = self.upsample2(out)
            out = self.deconv8(out)
            output_pred.append(out.unsqueeze(2))
        
        output_pred = torch.concat(output_pred,dim=2)
        final_output = torch.concat([output_prev.unsqueeze(1), output_pred.unsqueeze(1)],dim=1)
        
        output_img = final_output[:,:,0,:]
        #output_img = F.tanh(output_img)
        output_seg = final_output[:,:,1,:].unsqueeze(2)
        #print(output_img.shape, output_seg.shape)
        return output_img, output_seg, None

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
