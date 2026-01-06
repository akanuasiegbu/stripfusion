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

class SpatialTemporalSampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_height, kernel_width, factor=3):
        """
        in_channels:  the in_channel
        out_channels:  the out_channel
        kernel_height: Size of the convolving kernel height
        kernel_weight: Size of the convolving kernel weight
        """
        super().__init__()
        # what is the input channel dimension of the network
        # Do I need a separte S_x for different scales, the number of input channels I believe depends on the scale looked at
        # self.deform_list = nn.ModuleList(DeformConv(out_channels, in_channels, kernel_height, kernel_width) for i in range(4))
        # self.final_deform = DeformConv(out_channels, in_channels, kernel_height, kernel_width, offset_gen=False)
        self.deform_list = nn.ModuleList()
        self.deform_list.append(DeformConv(in_channels*2, 256, kernel_height, kernel_width))
        self.deform_list.append(DeformConv(256, 256, kernel_height, kernel_width))
        self.deform_list.append(DeformConv(256, 256, kernel_height, kernel_width))
        # self.deform_list.append(DeformConv(1024, 1024, kernel_height, kernel_width))
        # our spatiotemporal sampling block consists of four 3 × 3 deformable convolutional layers each with 1024 output channels
        self.final_deform = DeformConv(in_channels, out_channels, kernel_height, kernel_width, bias=True, last_layer=True)
       #######################################################################################################################################
        self.S_x = nn.Sequential(nn.Conv2d(out_channels, out_channels*factor, 1), nn.ReLU(True), 
                                 nn.Conv2d(out_channels*factor, out_channels*factor, 3, padding=1), nn.ReLU(True), 
                                 nn.Conv2d(out_channels*factor, out_channels, 1), nn.ReLU(True))
        self.softmax = nn.Softmax(dim=1) #if feat is B,F,C,H,W

        self.bn_list = nn.ModuleList()
        self.bn_list.append(nn.BatchNorm2d(256))
        self.bn_list.append(nn.BatchNorm2d(256))
        self.bn_list.append(nn.BatchNorm2d(256))
        self.bn_final = nn.BatchNorm2d(out_channels)
        self.bn_S_x =  nn.BatchNorm2d(out_channels)
        
    def forward(self, inputs):
        B, C, F, H, W = inputs.shape
        feat_t_and_t_plus_k = self.concat_features(inputs)
        
        for index, deform in enumerate(self.deform_list):
            # print(index)
            # can flatten down an pass all through at the same time
            feat_t_and_t_plus_k, offset_t_and_t_plus_k = deform(feat_t_and_t_plus_k)
            feat_t_and_t_plus_k = self.bn_list[index](feat_t_and_t_plus_k)

        #this is wrong because out is not f_t_k but f_t_and_t_
        f_t_and_t_plus_k = inputs.permute(0,2,1,3,4).contiguous().view(B*F, C,H,W)
        final_feat_t_and_t_plus_k, _ = self.final_deform(f_t_and_t_plus_k, feat_t_and_t_plus_k)
        final_feat_t_and_t_plus_k = self.bn_final(final_feat_t_and_t_plus_k)
        
        S_g_t_and_t_plus_k = self.S_x(final_feat_t_and_t_plus_k) #shape is (B*F, new_channel, H,W)
        S_g_t_and_t_plus_k = self.bn_S_x(S_g_t_and_t_plus_k)

        weight = self.calc_weights_for_feature_aggregation(S_g_t_and_t_plus_k.view(B,F,-1,H,W))
        weight_after_softmax = self.softmax(weight) ##softmax on frames dim, B,F,C,H,W
        # weight_after_softmax = torch.nn.functional.softmax(weight, dim=1, dtype=torch.float16) ##softmax on frames dim, B,F,C,H,W

        aggregated_features = torch.mul(weight_after_softmax, final_feat_t_and_t_plus_k.view(B, F,-1,H,W)) #element wise multiplication
        # aggregated_features = torch.sum(aggregated_features, dim=1, dtype=torch.float16)
        aggregated_features = aggregated_features.sum(dim=1) #sum up on frames dim
       
        # now I can go through a input into frame
        # Need to go through one combination
        return aggregated_features

    def calc_weights_for_feature_aggregation(self, feat):
        B,F,C,H,W = feat.shape
        S_g_t_t = feat[:,0,:,:,:] #last_frame
        S_g_t_t = S_g_t_t.unsqueeze(1).repeat(1, F,1,1,1) # expands the lastframe to match F dim
        S_g_t_t = S_g_t_t.reshape(B*F, C, H,W)
        S_g_t_t_norm = S_g_t_t / ( torch.norm(S_g_t_t, p=2, dim=1)[:, None, :, :] + 1e-6)
        feat_reshaped = feat.reshape(B*F, C,H,W)
        feat_norm = feat_reshaped / (torch.norm(feat_reshaped, p=2, dim=1)[:, None,:,:] + 1e-6)
        
        pixel_wise_cosine_similarity_v2 = S_g_t_t_norm *feat_norm
         
        w_t_t_k = torch.exp_(pixel_wise_cosine_similarity_v2)
        w_t_t_k = w_t_t_k.reshape(B, F, C, H, W)
        return w_t_t_k

    # def calc_weights_for_feature_aggregation(self, feat):
    #     B,F,C,H,W = feat.shape
    #     # if feat.isnan().any() or feat.isinf().any():
    #     #     raise RuntimeError("ERROR: Got NaN in feat {}".format(datetime.now()))
    #     S_g_t_t = feat[:,0,:,:,:] #last_frame
    #     S_g_t_t = S_g_t_t.unsqueeze(1).repeat(1, F,1,1,1) # expands the lastframe to match F dim
    #     if S_g_t_t.isnan().any() or S_g_t_t.isinf().any():
    #         import pdb; pdb.set_trace()
    #         # raise RuntimeError("ERROR: Got NaN in S_g_t_t {}".format(datetime.now()))
        
    #     if feat.isnan().any() or feat.isinf().any():
    #         import pdb; pdb.set_trace()
    #         # raise RuntimeError("ERROR: Got NaN in feat {}".format(datetime.now()))

    #     top_cosine = torch.mul(S_g_t_t, feat)

    #     if top_cosine.isnan().any() or top_cosine.isinf().any():
    #         import pdb; pdb.set_trace()
    #         # raise RuntimeError("ERROR: Got NaN in top_cosine {}".format(datetime.now()))

    #     bottom_cosine = torch.mul(torch.abs(S_g_t_t),torch.abs(feat)) #elment wise absolute value and then multiplication

    #     if bottom_cosine.isnan().any() or bottom_cosine.isinf().any():
    #         import pdb; pdb.set_trace()
    #         # raise RuntimeError("ERROR: Got NaN in bottom_cosine 1 {}".format(datetime.now()))
    #     eps = nn.Parameter(torch.full(bottom_cosine.shape, fill_value=1e-6, device="cuda", requires_grad=False), requires_grad=False)
    #     bottom_cosine = torch.maximum(bottom_cosine, eps) #element wise max, to prevent division by zero

    #     pixel_wise_cosine_similarity = torch.div(top_cosine, bottom_cosine)
    #     import pdb; pdb.set_trace()

    #     if pixel_wise_cosine_similarity.isnan().any() or pixel_wise_cosine_similarity.isinf().any():
    #         import pdb; pdb.set_trace()
    #         # raise RuntimeError("ERROR: Got NaN in pixel_wise_cosine_similarity {}".format(datetime.now()))
    #     w_t_t_k = torch.exp_(pixel_wise_cosine_similarity)
    #     return w_t_t_k
        

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