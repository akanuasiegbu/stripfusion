import torch
import torch.nn as nn
from  torchvision.ops  import deform_conv2d
from datetime import datetime

class OffsetGenerator(nn.Module):
    def __init__(self, ch, offset_groups, kernel_height, kernel_width):
        super().__init__()
        # what are the offset groups doing
        self.generator = nn.Sequential( nn.Conv2d(ch, 2*offset_groups*kernel_height*kernel_width, 3, padding=1), nn.ReLU()) 
        # self.generator =  nn.Conv2d(ch, 2*offset_groups*kernel_height*kernel_width, 3)
    def forward(self, inputs):
        # Desired ouput -> Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width, out_height, out_width])
        return self.generator(inputs)
class DeformConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_height, kernel_width, bias=False, last_layer=False):
        """
        in_channels: Number of channels in the input image
        out_channels: Number of channels in the output image
        kernel_height: Size of the convolving kernel height
        kernel_weight: Size of the convolving kernel weight
        last_layer: True
        """
        super().__init__()
        # weights shape -> (Tensor[out_channels, in_channels // groups, kernel_height, kernel_width])
        self.last_layer = last_layer
        if not last_layer:
            self.offsets = OffsetGenerator(in_channels, 1, kernel_height, kernel_width)
            # self.bias = None
        else:
            self.offsets = OffsetGenerator(256,1,kernel_height, kernel_width)
            # self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_height, kernel_width))
        self.bias = nn.Parameter(torch.Tensor(out_channels)) if bias else None
        self.initialize_weights()
        
    def initialize_weights(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')     
    def forward(self, inputs, offsets_inputs=None):
        # should add assertions to make sure shape match
        if not self.last_layer:
            offsets = self.offsets(inputs)
        else:
            offsets = self.offsets(offsets_inputs)
        # self.bias = self.bias if self.bias is None else self.bias.to(inputs.dtype)
        return deform_conv2d(inputs, offsets, self.weight.to(inputs.dtype), bias=self.bias if self.bias is None else self.bias.to(inputs.dtype), padding=1), offsets

class SpatialTemporalSamplingv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_height, kernel_width, factor=3):
        """
        in_channels:  the in_channel
        out_channels:  the out_channel
        kernel_height: Size of the convolving kernel height
        kernel_weight: Size of the convolving kernel weight
        """
        super().__init__()
        self.deform_list = nn.ModuleList()
        self.deform_list.append(DeformConv(in_channels*2, 256, kernel_height, kernel_width))
        self.deform_list.append(DeformConv(256, 256, kernel_height, kernel_width))
        self.final_deform = DeformConv(in_channels, out_channels, kernel_height, kernel_width, bias=True, last_layer=True)
       
        self.relu = nn.ReLU()
        self.bn_list = nn.ModuleList()
        self.bn_list.append(nn.BatchNorm2d(256))
        self.bn_list.append(nn.BatchNorm2d(256))
        self.spatial_mlp = nn.Conv2d(out_channels*3, out_channels, 1) #would probably want to do 256xframes instead of 3 so that can adapt to longer frames
        
    def forward(self, inputs):
        B, C, F, H, W = inputs.shape
        feat_t_and_t_plus_k = self.concat_features(inputs)
        
        for index, deform in enumerate(self.deform_list):
            # print(index)
            # can flatten down an pass all through at the same time
            feat_t_and_t_plus_k, offset_t_and_t_plus_k = deform(feat_t_and_t_plus_k)
            feat_t_and_t_plus_k = self.relu(feat_t_and_t_plus_k)
            feat_t_and_t_plus_k = self.bn_list[index](feat_t_and_t_plus_k)
        
        
        #this is wrong because out is not f_t_k but f_t_and_t_
        f_t_and_t_plus_k = inputs.permute(0,2,1,3,4).contiguous().view(B*F, C,H,W)
        final_feat_t_and_t_plus_k, _ = self.final_deform(f_t_and_t_plus_k, feat_t_and_t_plus_k)
        feat_t_and_t_plus_k = final_feat_t_and_t_plus_k
        
        feat_t_and_t_plus_k = feat_t_and_t_plus_k.view(B, -1, H, W)
        # import pdb; pdb.set_trace()
        feat_t_and_t_plus_k= self.spatial_mlp(feat_t_and_t_plus_k) #acts like weight for frames to aggreagte so learned weight still but linear and spt


        return feat_t_and_t_plus_k
    


    def concat_features(self, inputs):
        B, C, F, H, W = inputs.shape
        inputs = inputs.permute(2,0,1,3,4).contiguous() #F, B, C, H, W 
        inputs_last = inputs[0]
        features_frames_combined = []
        for inputs_frame in inputs:
            features_frames_combined.append(torch.cat((inputs_last, inputs_frame), 1))
        out = torch.stack(features_frames_combined).permute(1,0,2,3,4).contiguous().view(B*F, 2*C, H, W) # Move to Batch*frames, 2C, H, W
        return out


if __name__ == '__main__':
    # dconv = DeformConv(50,50,3,3)
    # rand = torch.rand(4,50,3,20,20)
    rand = torch.rand(4,1024,3,20,20)
    # ouput = dconv(rand)
    stnet = SpatialTemporalSampling(1024, 27, 3,3)
    out = stnet(rand)
    print(out.shape)