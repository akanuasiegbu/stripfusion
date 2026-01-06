import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange, reduce, asnumpy, parse_shape
from einops.layers.torch import Rearrange, Reduce
import math

class MlpBlock(nn.Module):
    def __init__(self, mlp_dim, pdrop):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.mlp1 = nn.Linear(mlp_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(pdrop)
        self.mlp2 = nn.Linear(mlp_dim, mlp_dim)

    def forward(self, x):
        y = self.mlp1(x)
        y = self.gelu(y)
        y = self.drop(y)
        return self.mlp2(y)
    
class SpatialMixerBlock(nn.Module):
    def __init__(self, channels_mlp_dim, rgb_ir_tokens_mlp_dim, pdrop):
        super().__init__()
        self.rgb_ir_spatial_mix = MlpBlock(rgb_ir_tokens_mlp_dim, pdrop) 
        self.channel_mix = MlpBlock(channels_mlp_dim, pdrop)
        # might make sense to either use larger tokens so that temporal relations are easier captured
        # could image that if we have a small window it would be hard to find 
        self.norm1  = nn.LayerNorm(channels_mlp_dim) #???
        self.norm2  = nn.LayerNorm(channels_mlp_dim) #???

    def forward(self, x):
        b, f ,p, s, c = x.shape #batch, frame, patch_num, token_mlp_dim, channels (Input Shape)

        y = self.norm1(x)
        y = rearrange(y, 'b f p s c -> b f p c s')
        y = self.rgb_ir_spatial_mix(y)
        y = rearrange(y, 'b f p c s-> b f p s c')
        x = x + y
        
        y = self.norm2(x)
        y = self.channel_mix(y)
        x = x + y  # b,f,p,s,c (Ouput Shape)
        return x

class TemporalMixerBlock(nn.Module):
    def __init__(self, channels_mlp_dim,  num_of_frames, pdrop):
        super().__init__()
        self.temporal_mix = MlpBlock(num_of_frames, pdrop)
        self.norm3  = nn.LayerNorm(channels_mlp_dim)
        
    def forward(self, x):
        # b, f ,p, s, c (Input Shape)
        y = self.norm3(x)
        y = rearrange(y, 'b f p s c -> b p s c f')
        y = self.temporal_mix(y)
        y = rearrange(y, 'b p s c f -> b f p s c')
        x = x + y # b, f ,p, s, c (Ouput Shape)
        return x


class TemporalMixerBlockv2(nn.Module):
    def __init__(self, channels_mlp_dim, num_patches, num_of_frames, pdrop):
        super().__init__()
        self.spatial_temporal_patch_mix = MlpBlock(num_patches*num_of_frames, pdrop)
        self.norm4 = nn.LayerNorm(channels_mlp_dim)
    
    def forward(self, x):
        b, f ,p, s, c = x.shape
        y = self.norm4(x)
        y = rearrange(y, 'b f p s c -> b c s (f p)')
        y = self.spatial_temporal_patch_mix(y)
        y = rearrange(y,' b c s (f p) -> b f p s c', f = f)
        x = x + y
        
        return x
        
        
class SpatialTemporalBlock(nn.Module):
    def __init__(self, channels_mlp_dim, rgb_ir_tokens_mlp_dim, num_temporal_mix_block, num_of_frames, pdrop):
        super().__init__()
        self.spatial_mix = SpatialMixerBlock(channels_mlp_dim, rgb_ir_tokens_mlp_dim, pdrop)
        self.temporal_mix = nn.Sequential(*[TemporalMixerBlock(channels_mlp_dim, num_of_frames, pdrop)
                                            for _ in range(num_temporal_mix_block)])
        
    def forward(self, initial_patches):
        y = self.spatial_mix(initial_patches)
        y = self.temporal_mix(y)
        return y

class SpatialTemporalBlockv2(nn.Module):
    def __init__(self, channels_mlp_dim, rgb_ir_tokens_mlp_dim, num_temporal_mix_block,
                 num_patches, num_of_frames, pdrop):
        super().__init__()
        self.spatial_mix = SpatialMixerBlock(channels_mlp_dim, rgb_ir_tokens_mlp_dim, pdrop)
        self.temporal_mix = nn.Sequential(*[TemporalMixerBlockv2(channels_mlp_dim, num_patches, num_of_frames, pdrop)
                                            for _ in range(num_temporal_mix_block)])
        
    def forward(self, initial_patches):
        y = self.spatial_mix(initial_patches)
        y = self.temporal_mix(y)
        return y

class TemporalBlockv2(nn.Module):
    def __init__(self, channels_mlp_dim, num_temporal_mix_block, num_patches, num_of_frames, pdrop):
        super().__init__()
        self.temporal_mix =  nn.Sequential(*[TemporalMixerBlockv2(channels_mlp_dim, num_patches, num_of_frames, pdrop)
                                            for _ in range(num_temporal_mix_block)])
    def forward(self, initial_patches):
        y = self.temporal_mix(initial_patches)
        return y
        
class BAMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_ir_tokens_mlp_dim = None
        self.mixerblock  = None

    def before_mixer(self, inputs):
        rgb_feat = inputs[0]
        ir_feat = inputs[1]
        b, c, f, h, w = rgb_feat.shape
        
        patch = int(math.sqrt(self.rgb_ir_tokens_mlp_dim/2))
        initial_patches_rgb = rearrange(rgb_feat, 'b c f (h ph) (w pw) -> b f (h w) (ph pw) c', ph = patch , pw = patch)
        initial_patches_ir = rearrange(ir_feat, 'b c f (h ph) (w pw) -> b f (h w) (ph pw) c', ph = patch , pw = patch)
        # import pdb; pdb.set_trace()
        initial_patches = torch.cat([initial_patches_rgb, initial_patches_ir], dim=3)  # concat
        return initial_patches, patch
    
    def after_mixer(self, output_patches, patch):
        y = rearrange(output_patches,' b f p s c-> b c f p s')
        dim = int(math.sqrt(y.shape[-2]))
        rgb_feat_fus = y[:,:,:,:,:self.rgb_ir_tokens_mlp_dim//2]
        ir_feat_fus = y[:,:,:,:,self.rgb_ir_tokens_mlp_dim//2:]
        rgb_feat_fus = rearrange(rgb_feat_fus, 'b c f (p1 p2) (ph pw)->b c f (p1 ph) (p2 pw)', p1=dim,p2=dim, ph=patch, pw=patch)
        ir_feat_fus = rearrange(ir_feat_fus, 'b c f (p1 p2) (ph pw)->b c f (p1 ph) (p2 pw)', p1=dim,p2=dim, ph=patch, pw=patch)

        return rgb_feat_fus, ir_feat_fus
    
    def forward(self, inputs):
        initial_patches, patch = self.before_mixer(inputs)
        output_patches = self.mixerblock(initial_patches)
        
        rgb_feat_fus, ir_feat_fus = self.after_mixer(output_patches, patch)

        return rgb_feat_fus, ir_feat_fus      


class MLPSpatialMixer(BAMixer):
    def __init__(self, channels_mlp_dim, rgb_ir_tokens_mlp_dim, num_spatial_block, pdrop=0.25):
        super().__init__()
        self.rgb_ir_tokens_mlp_dim = rgb_ir_tokens_mlp_dim
        self.mixerblock = nn.Sequential(*[SpatialMixerBlock(channels_mlp_dim, rgb_ir_tokens_mlp_dim, pdrop) 
                                                                for _ in range(num_spatial_block) ])


class MLPSpatialTemporalMixer(BAMixer):
    def __init__(self, channels_mlp_dim, rgb_ir_tokens_mlp_dim, num_st_block, 
                 num_temporal_mix_block, num_of_frames, pdrop=0.25): # frames last 
        super().__init__()
        self.rgb_ir_tokens_mlp_dim = rgb_ir_tokens_mlp_dim
        self.mixerblock = nn.Sequential(*[SpatialTemporalBlock(channels_mlp_dim, rgb_ir_tokens_mlp_dim,
                                                                num_temporal_mix_block, num_of_frames, pdrop) 
                                                                for _ in range(num_st_block) ])
            

class MLPSpatialTemporalMixerv2(BAMixer):
    def __init__(self, channels_mlp_dim, rgb_ir_tokens_mlp_dim, num_st_block, 
                 num_temporal_mix_block, num_patches, num_of_frames, pdrop=0.25): # frames last 
        # super().__init__(channels_mlp_dim, rgb_ir_tokens_mlp_dim, num_st_block, num_temporal_mix_block, num_of_frames)
        super().__init__()
        self.rgb_ir_tokens_mlp_dim = rgb_ir_tokens_mlp_dim
        self.mixerblock = nn.Sequential(*[SpatialTemporalBlockv2(channels_mlp_dim, rgb_ir_tokens_mlp_dim, 
                                                                    num_temporal_mix_block, num_patches,
                                                                    num_of_frames, pdrop) 
                                                                    for _ in range(num_st_block) ])

class SpatialMixerConcatBlock(nn.Module):
    def __init__(self, channels_mlp_dim, rgb_ir_tokens_mlp_dim, pdrop):
        super().__init__()
        self.rgb_ir_spatial_mix = MlpBlock(rgb_ir_tokens_mlp_dim, pdrop) 
        self.channel_mix = MlpBlock(channels_mlp_dim, pdrop)
        # might make sense to either use larger tokens so that temporal relations are easier captured
        # could image that if we have a small window it would be hard to find 
        self.norm1  = nn.LayerNorm(channels_mlp_dim) #???
        self.norm2  = nn.LayerNorm(channels_mlp_dim) #???

    def forward(self, x):
        b ,p, s, c = x.shape #batch, patch_num, token_mlp_dim, channels (Input Shape)

        y = self.norm1(x)
        y = rearrange(y, 'b p s c -> b p c s')
        y = self.rgb_ir_spatial_mix(y)
        y = rearrange(y, 'b p c s-> b p s c')
        x = x + y
        
        y = self.norm2(x)
        y = self.channel_mix(y)
        x = x + y  # b,p,s,c (Ouput Shape)
        return x

class BAConcatMixer(BAMixer):
    def before_mixer(self, inputs):
        rgb_feat = inputs[0]
        ir_feat = inputs[1]
        b, c, h, w = rgb_feat.shape
        
        patch = int(math.sqrt(self.rgb_ir_tokens_mlp_dim/2))
        initial_patches_rgb = rearrange(rgb_feat, 'b c (h ph) (w pw) -> b (h w) (ph pw) c', ph = patch , pw = patch)
        initial_patches_ir = rearrange(ir_feat, 'b c (h ph) (w pw) -> b (h w) (ph pw) c', ph = patch , pw = patch)
        initial_patches = torch.cat([initial_patches_rgb, initial_patches_ir], dim=2)  # concat
        return initial_patches, patch
    
    def after_mixer(self, output_patches, patch):
        y = rearrange(output_patches,' b p s c-> b c p s')
        dim = int(math.sqrt(y.shape[-2]))
        rgb_feat_fus = y[:,:,:,:self.rgb_ir_tokens_mlp_dim//2]
        ir_feat_fus = y[:,:,:,self.rgb_ir_tokens_mlp_dim//2:]
        rgb_feat_fus = rearrange(rgb_feat_fus, 'b c (p1 p2) (ph pw)->b c (p1 ph) (p2 pw)', p1=dim,p2=dim, ph=patch, pw=patch)
        ir_feat_fus = rearrange(ir_feat_fus, 'b c (p1 p2) (ph pw)->b c (p1 ph) (p2 pw)', p1=dim,p2=dim, ph=patch, pw=patch)

        return rgb_feat_fus, ir_feat_fus
    
class MLPSpatialConcatMixer(BAConcatMixer):
    def __init__(self, channels_mlp_dim, rgb_ir_tokens_mlp_dim, num_spatial_block, pdrop=0.25):
        super().__init__()
        self.rgb_ir_tokens_mlp_dim = rgb_ir_tokens_mlp_dim
        self.mixerblock = nn.Sequential(*[SpatialMixerConcatBlock(channels_mlp_dim, rgb_ir_tokens_mlp_dim, pdrop) 
                                                                for _ in range(num_spatial_block) ])

class RectBAMix(BAMixer):
    # main difference has to do with patch sizes
    def __init__(self):
        super().__init__()
        self.patch_h = None
        self.patch_w = None
        self.dim_h = None
        self.dim_w = None
        self.rgb_ir_tokens_mlp_dim = None
        self.mixerblock  = None

    def before_mixer(self, inputs):
        rgb_feat = inputs[0]
        ir_feat = inputs[1]
        b, c, f, h, w = rgb_feat.shape
        
        patch_h = self.patch_h[self.rgb_ir_tokens_mlp_dim]
        patch_w = self.patch_w[self.rgb_ir_tokens_mlp_dim]
        initial_patches_rgb = rearrange(rgb_feat, 'b c f (h ph) (w pw) -> b f (h w) (ph pw) c', ph = patch_h , pw = patch_w)
        initial_patches_ir = rearrange(ir_feat, 'b c f (h ph) (w pw) -> b f (h w) (ph pw) c', ph = patch_h , pw = patch_w)
        initial_patches = torch.cat([initial_patches_rgb, initial_patches_ir], dim=3)  # concat
        return initial_patches, patch_h, patch_w
    
    def after_mixer(self, output_patches, patch_h, patch_w):
        y = rearrange(output_patches,' b f p s c-> b c f p s')
        dim_h = self.dim_h[self.rgb_ir_tokens_mlp_dim]
        dim_w = self.dim_w[self.rgb_ir_tokens_mlp_dim]
        rgb_feat_fus = y[:,:,:,:,:self.rgb_ir_tokens_mlp_dim//2]
        ir_feat_fus = y[:,:,:,:,self.rgb_ir_tokens_mlp_dim//2:]
        rgb_feat_fus = rearrange(rgb_feat_fus, 'b c f (p1 p2) (ph pw)->b c f (p1 ph) (p2 pw)', p1=dim_h,p2=dim_w, ph=patch_h, pw=patch_w)
        ir_feat_fus = rearrange(ir_feat_fus, 'b c f (p1 p2) (ph pw)->b c f (p1 ph) (p2 pw)', p1=dim_h,p2=dim_w, ph=patch_h, pw=patch_w)

        return rgb_feat_fus, ir_feat_fus

    def forward(self, inputs):
        initial_patches, patch_h, patch_w = self.before_mixer(inputs)
        output_patches = self.mixerblock(initial_patches)
        
        rgb_feat_fus, ir_feat_fus = self.after_mixer(output_patches, patch_h, patch_w)

        return rgb_feat_fus, ir_feat_fus      


class MLPSpatialTemporalRectMixerv2(RectBAMix):
    def __init__(self, channels_mlp_dim, rgb_ir_tokens_mlp_dim, num_st_block, 
                 num_temporal_mix_block, num_patches, num_of_frames, pdrop=0.25,
                 patch_h=None, patch_w = None, dim_h = None, dim_w = None): # frames last 
        super().__init__()
        self.patch_h = {800: 20, 400:20 ,200: 20 } 
        self.patch_w = {800: 20, 400:10, 200:5} # patch sizes square -> rectangle at smaller feature levels
        self.dim_h = {800:4, 400:2 , 200: 1}  # dim_h * dim_w = num_patches, order of dim_h and dim_w matter
        self.dim_w = {800: 4, 400:4 ,200: 4 }
        self.rgb_ir_tokens_mlp_dim = rgb_ir_tokens_mlp_dim
        self.mixerblock = nn.Sequential(*[SpatialTemporalBlockv2(channels_mlp_dim, rgb_ir_tokens_mlp_dim, 
                                                                    num_temporal_mix_block, num_patches,
                                                                    num_of_frames, pdrop) 
                                                                    for _ in range(num_st_block) ])

class MLPTemporalMixerv2(BAMixer):
    def __init__(self, channels_mlp_dim, rgb_ir_tokens_mlp_dim,
                num_temporal_mix_block=2, num_patches=16, num_of_frames=3, pdrop=0.1):
        super().__init__()
        self.rgb_ir_tokens_mlp_dim = rgb_ir_tokens_mlp_dim
        self.mixerblock = TemporalBlockv2(channels_mlp_dim, num_temporal_mix_block, num_patches,num_of_frames, pdrop)