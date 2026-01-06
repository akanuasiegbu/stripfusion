import torch
import torch.nn as nn
import torch.nn.functional as FF
import numpy as np
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
import models.fusion.MLPmixer as MLPmixer 

class BN_Activ_Conv(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, stride=(1, 1, 1), dilation=(1, 1, 1), groups=1):
        super().__init__()
        self.BN = nn.BatchNorm3d(out_channels)
        self.Activation = activation
        padding = [int((dilation[j] * (kernel_size[j] - 1) - stride[j] + 1) / 2) for j in range(3)]  # Same padding
        self.Conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=groups, bias=False)

    def forward(self, img):
        img = self.BN(img)
        img = self.Activation(img)
        img = self.Conv(img)
        return img

class DepthWise_Conv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_merge = BN_Activ_Conv(channels, nn.GELU(), channels, (1,3, 3), groups=channels)

    def forward(self, img):
        img = self.conv_merge(img)
        return img


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class RelativePosition(nn.Module):
#Adapt for 2H, W 

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)
        trunc_normal_(self.embeddings_table, std=.02)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat)
        embeddings = self.embeddings_table[final_mat]

        return embeddings
    
class StripMLPFusion_Block(nn.Module):
    def __init__(self, channels, H, W):
        super().__init__()
        # assert W == H
        self.channels = channels
        self.activation = nn.GELU()
        self.BN = nn.BatchNorm3d(channels//2)

        if channels % 80 == 0:
            patch = 2
        else:
            patch = 4

        self.ratio = 1
        self.C = int(channels *0.5/ patch)
        self.chan = self.ratio * self.C

        self.proj_h = nn.Conv3d(H*self.C, self.chan*H, (1, 1, 5), stride=1, padding=(0, 0, 5//2), groups=self.C,bias=True)
        self.proj_w = nn.Conv3d(self.C*W*2, self.chan*W*2, (1, 1, 5), stride=1, padding=(0, 0, 5//2), groups=self.C, bias=True) # W*2

        self.fuse_h = nn.Conv3d(channels, channels//2, (1,1,1), (1,1,1), bias=False)
        self.fuse_w = nn.Conv3d(channels, channels//2, (1,1,1), (1,1,1), bias=False)

        self.mlp=nn.Sequential(nn.Conv3d(channels, channels, 1, 1,bias=True),nn.BatchNorm3d(channels),nn.GELU())
        self.mlp_ir=nn.Sequential(nn.Conv3d(channels, channels, 1, 1,bias=True),nn.BatchNorm3d(channels),nn.GELU())


        dim = channels // 2

        self.fc_h = nn.Conv3d(dim, dim, (1,5,7), stride=1, padding=(0, 5//2, 7//2), groups=dim, bias=False) 
        self.fc_w = nn.Conv3d(dim, dim, (1,7,5), stride=1, padding=(0, 7//2, 5//2), groups=dim, bias=False)

        self.reweight = Mlp(dim, dim // 2, dim * 3)

        self.fuse = nn.Conv3d(channels, channels, (1, 1,1), (1, 1,1), bias=False)

        self.relate_pos_h = RelativePosition(channels//2, H*2)
        self.relate_pos_w = RelativePosition(channels//2, W*2)
        
    def forward(self, x_rgb, x_ir):
        B,C, F, H, W = x_rgb.shape
        H1 = 2*H
        # x_rgb = x[0]
        # x_ir = x[1]
        
        x_rgb = self.mlp(x_rgb)
        x_ir = self.mlp_ir(x_ir)
        
        # https://pytorch.org/docs/stable/notes/autograd.html
        rgb_ir = torch.zeros((B, C, F, H1, W),
                              device=x_rgb.device, dtype=x_rgb.dtype)
                             #, dtype=torch.float16).cuda()
        rgb_ir[:,:,:,::2, :] = x_rgb
        rgb_ir[:,:,:,1::2, :] = x_ir
        
        rgb_ir_1 = rgb_ir[:, :C//2, :, :, :]
        rgb_ir_2 = rgb_ir[:,C//2:, :, :, :]
        
        rgb_ir_1  = self.strip_mlp_rgb_ir(rgb_ir_1)
        
        x_w = self.fc_h(rgb_ir_2)
        x_h = self.fc_w(rgb_ir_2)
        
        
        att = FF.adaptive_avg_pool3d(x_h + x_w + rgb_ir_2, output_size=(F,1,1))
        
        att = self.reweight(att).reshape(B, C//2, 3, F).permute(2,0,1,3).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        
        rgb_ir_2 = x_h * att[0] + x_w * att[1] + rgb_ir_2 * att[2]
        
        x = self.fuse(torch.cat([rgb_ir_1, rgb_ir_2], dim=1))
        
        return x[:,:,:,::2,:], x[:,:,:,1::2,:] #rgb, thermal
        
    def strip_mlp_rgb_ir(self, rgb_ir):
        B, C, F, H1, W =  rgb_ir.shape
        pos_h = self.relate_pos_h(H1, W).unsqueeze(0).permute(0, 3, 1, 2).unsqueeze(2)
        pos_w = self.relate_pos_w(H1, W).unsqueeze(0).permute(0, 3, 1, 2).unsqueeze(2)
        C1 = int(C/self.C)
        
        rgb_ir_pos = rgb_ir + pos_h
        x_h = rgb_ir_pos.view(B, C1, self.C, F, H1, W )
        
        x_h = x_h.permute(0, 1, 3, 4, 2, 5 ).contiguous().view(B, C1, F, H1, self.C*W)
        
        x_h = self.proj_h(x_h.permute(0, 4, 2, 1, 3)).permute(0, 3, 2, 4, 1) # cross modality spatial and channel

        x_h = x_h.view(B, C1, F, H1, self.C, W).permute(0, 1, 4, 2, 3, 5).contiguous().view(B, C, F, H1, W)
        
        x_h = self.fuse_h(torch.cat([x_h, rgb_ir], dim=1))
        x_h = self.activation(self.BN(x_h)) + pos_w
        
        x_w = self.proj_w(x_h.view(B, C1, self.C, F, H1, W).permute(0,2,4,3,1,5).contiguous().view(B, self.C*H1, F, C1, W)) #cross modality in channel 
        x_w = x_w.contiguous().view(B, self.C, H1, F, C1, W).permute(0, 4, 1, 3, 2, 5).contiguous().view(B, C, F, H1, W)

        rgb_ir = self.fuse_w(torch.cat([rgb_ir, x_w], dim = 1))
        
        return rgb_ir
    
class TokenMixing(nn.Module):
    r""" Token mixing of Strip MLP

    Args:
    """

    def __init__(self, C, H, W):
        super().__init__()
        self.smlp_block = StripMLPFusion_Block(C, H, W)
        self.dwsc = DepthWise_Conv(C)
    
    def forward(self, x_rgb, x_ir):
        x_rgb = self.dwsc(x_rgb)
        x_ir = self.dwsc(x_ir)
        x_rgb, x_ir = self.smlp_block(x_rgb, x_ir)

        return x_rgb, x_ir

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2,3), keepdim=True) # h,w dim
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ChannelMixing(nn.Module):

    def __init__(self, in_channel, alpha, use_dropout=False, drop_rate=0):
        super().__init__()

        self.use_dropout = use_dropout

        self.conv_77 = nn.Conv3d(in_channel, in_channel, (1, 11, 11), 1, (0, 5, 5), groups=in_channel, bias=False)
        self.layer_norm = nn.LayerNorm(in_channel)
        self.fc1 = nn.Linear(in_channel, alpha * in_channel)
        self.activation = nn.GELU()
        self.drop = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(alpha * in_channel, in_channel)

        self.grn = GRN(3*in_channel)

    
    def forward(self, x_rgb, x_ir):
        B,C, F, H, W = x_rgb.shape
        H1 = 2*H
        
        
        # https://pytorch.org/docs/stable/notes/autograd.html
        rgb_ir = torch.zeros((B, C, F, H1, W),
                              device=x_rgb.device, dtype=x_rgb.dtype)
                             #, dtype=torch.float16).cuda()
        rgb_ir[:,:,:,::2, :] = x_rgb
        rgb_ir[:,:,:,1::2, :] = x_ir

        x = rgb_ir.contiguous()
        
        x = self.conv_77(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.layer_norm(x)
        
        x = self.fc1(x)
        x = self.activation(x)
        x = self.grn(x)

        x = self.fc2(x)

        x = x.permute(0, 4, 1, 2, 3)

        return x[:,:,:,::2,:] , x[:,:,:,1::2,:]
    
class BasicBlock(nn.Module):
    def __init__(self, in_channel, H, W, alpha, use_dropout=False, drop_rate=0):
        super().__init__()

        self.token_mixing = TokenMixing(in_channel, H, W)
        self.channel_mixing = ChannelMixing(in_channel, alpha, use_dropout, drop_rate)
        
        drop_rate = 0.1

        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x_rgb, x_ir):
        rgb, ir =  self.token_mixing(x_rgb, x_ir) 
        B, C, F, H, W = rgb.shape
        
        concat_rgb_ir_drop_1 = self.drop_path(torch.cat((rgb, ir), dim=1))
        rgb = concat_rgb_ir_drop_1[:,:C,:,:,:]
        ir = concat_rgb_ir_drop_1[:,C:,:,:,:]
        
        x_rgb = x_rgb + rgb
        x_ir = x_ir + ir
        
        rgb1, ir1 = self.channel_mixing(x_rgb, x_ir)
        concat_rgb_ir_drop_2 = self.drop_path(torch.cat((rgb1, ir1), dim=1))
        
        
        rgb2 = concat_rgb_ir_drop_2[:,:C,:,:,:]
        ir2 = concat_rgb_ir_drop_2[:,C:,:,:,:]
        
        rgb1 = x_rgb + rgb2
        ir1 =  x_ir + ir2

        return rgb1, ir1

class StripMLPNet(nn.Module):
    """
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        layers (tuple(int)): Depth of each Swin Transformer layer.
        drop_rate (float): Dropout rate. Default: 0
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
    """

    def __init__(self, embed_dim, H, W, layers= 1, drop_rate=0.5,
                 norm_layer=nn.BatchNorm3d, alpha=3, use_dropout=False, patch_norm=True, **kwargs):
        super().__init__()

        self.embed_dim = embed_dim
        self.drop_rate = drop_rate


        self.blocks1 = nn.ModuleList()
        for i in range(layers):
            basic = BasicBlock(embed_dim, H, W, alpha, use_dropout=use_dropout, drop_rate=drop_rate)
            self.blocks1.append(basic)


    def forward(self, x_rgb, x_ir):
        for b in self.blocks1:
            x_rgb, x_ir = b(x_rgb, x_ir)
        return x_rgb, x_ir

class StripMLPTemporalFusionBlockv2(nn.Module):
    def __init__(self, embed_dim, H, W, channels_mlp_dim, rgb_ir_tokens_mlp_dim, numframes):
        super().__init__()
        self.strip_fusion = StripMLPNet(embed_dim, H, W)
        self.temporal_fus = MLPmixer.MLPTemporalMixerv2(channels_mlp_dim, rgb_ir_tokens_mlp_dim, num_of_frames=numframes)
    def forward(self, rgb, ir):
        x_rgb, x_ir = self.strip_fusion(rgb, ir)
        rgb_out, ir_out = self.temporal_fus(torch.stack((x_rgb, x_ir)))
        return rgb_out, ir_out

class StripMLPMixer(nn.Module):
    def __init__(self, embed_dim, height, num_spat_blocks):
        super().__init__()
        width = height
        self.stripmlp = nn.ModuleList([ StripMLPNet(embed_dim, height,width)
                                       for _ in range(num_spat_blocks)
                                       ])
    def forward(self, inputs):
        rgb, ir = inputs[0], inputs[1]
        for b in self.stripmlp:
            rgb, ir = b(rgb, ir)
        return rgb, ir
    
class StripMLPTemporalMixerv2(nn.Module): #This works
    def __init__(self, embed_dim, height, rgb_ir_tokens_mlp_dim, num_st_blocks, numframes):
        super().__init__()
        channels_mlp_dim = embed_dim
        width = height

        self.st_block = nn.ModuleList([StripMLPTemporalFusionBlockv2(embed_dim, height, width, channels_mlp_dim, rgb_ir_tokens_mlp_dim, numframes)
                                       for _ in range(num_st_blocks)
                                       ])

    def forward(self, inputs, use_tadaconv=True, numframes=None):
        rgb, ir = inputs[0], inputs[1]

        # Reshape 4D tensors to 5D 
        if not use_tadaconv:
            fr = numframes
            bf, c, h, w = rgb.shape
            bs = bf // fr
            rgb = rgb.reshape(bs, fr, c, h, w).permute(0, 2, 1, 3, 4)
            ir = ir.reshape(bs, fr, c, h, w).permute(0, 2, 1, 3, 4)

        for b in self.st_block:
            rgb, ir = b(rgb,ir)
        
        # Reshape back to 4D 
        if not use_tadaconv:
            #B, C, F, H, W -> B*F, C, H, W
            rgb = rgb.permute(0, 2, 1, 3, 4).reshape(bs*fr, c, h, w)
            ir = ir.permute(0, 2, 1, 3, 4).reshape(bs*fr, c, h, w)
        
        return rgb, ir


if __name__ == "__main__":
    import os
    H, W = 80, 80
    C = 256
    rgb = torch.ones((10, C, 3, H, W))
    ir = torch.ones((10, C , 3, H, W))*4
    

    # stripmlp = StripMLPFusion_Block(channels= C, H=H, W=W)
    # rgb_1, ir_1 = stripmlp(rgb, ir)
    # total_parms = sum(param.numel() for param in stripmlp.parameters())
    
    # stripmlpnet =  StripMLPNet(embed_dim=C, H=H, W=W)
    # rgb_2, ir_2 = stripmlpnet(rgb, ir)
    # total_parms = sum(param.numel() for param in stripmlpnet.parameters())
    
    stripmlptemporal = StripMLPTemporalMixerv2(C, H,W, C, 400)
    rgb_3, ir_3 = stripmlptemporal(rgb, ir)
    total_parms = sum(param.numel() for param in stripmlptemporal.parameters())
    print(total_parms)
