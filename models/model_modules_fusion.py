import torch.nn as nn
import torch.nn.functional as F
import torch
import models.model_modules_shared as model_modules_shared
from models.tadaconv_v2 import TAdaConv2dV2
from einops import rearrange, reduce, asnumpy, parse_shape
from einops.layers.torch import Rearrange, Reduce
import math
from models.deformable_attention import deformable_attn_pytorch, \
    LearnedPositionalEncoding, constant_init, xavier_init
import warnings
import math


class DeformableSpatialAttentionLayer(nn.Module):
    def __init__(self, 
                 embed_dims,
                 num_heads=8,
                 num_points=12,
                 dropout=0.1):
        super(DeformableSpatialAttentionLayer, self).__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        
        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        # however, CUDA is not available in this implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.'
                'However, CUDA is not available in this implementation.')
            
        assert dim_per_head % 2 == 0, "embed_dims must be divisible by 2"
        
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points
        self.dropout = nn.Dropout(dropout)
        self.sampling_offsets = nn.Linear(self.embed_dims, num_heads * num_points * 2)
        self.attention_weights = nn.Linear(self.embed_dims, num_heads * num_points)
        self.value_proj = nn.Linear(self.embed_dims, dim_per_head)
        self.output_proj = nn.Linear(dim_per_head, self.embed_dims)
        self.init_weights()
    
    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, 2).repeat(1, 1, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        # TODO: Remove the hard coded half precision
        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        
    def forward(self, 
                query,
                key=None,
                value=None,
                query_pos=None,
                identity=None,
                device='cuda',
                dtype=torch.half,
                spatial_shapes=None,):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                (bs, num_query, embed_dims).
            value (Tensor): The value tensor with shape
                (bs, num_query, embed_dims).
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            spatial_shapes (tuple): Spatial shape of features (h, w).

        Returns:
             Tensor: forwarded results with shape [bs, num_query, embed_dims].
        """
        
        # import pdb; pdb.set_trace()
        
        bs, num_query, embed_dims = query.shape
        h, w = spatial_shapes
        
        if identity is None:
            identity = query
        
        if query_pos is not None:
            query = query + query_pos
        value = self.value_proj(value)
        # if key_padding_mask is not None:
        #     value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.reshape(bs, num_query, self.num_heads, -1) # bs, num_query, num_heads, embed_dims//num_heads
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(bs, num_query, self.num_heads, self.num_points, 2) # bs, num_query, num_heads, num_points, 2
        attention_weights = self.attention_weights(query).view(bs, num_query, self.num_heads, self.num_points) # bs, num_query, num_heads, num_points
        attention_weights = attention_weights.softmax(-1).to(dtype) # TODO: attention_weights.softmax(-1) changed attention_weights from half to float
        
        reference_points = self.get_reference_points(h, w, bs=bs, device=device, dtype=dtype) # bs, num_query, 2
        offset_normalizer = torch.Tensor([w, h]).to(device).to(dtype)
        sampling_locations = reference_points[:, :, None, None, :] \
            + sampling_offsets / offset_normalizer
        # if reference_points.shape[-1] == 2:
        #     offset_normalizer = torch.stack(
        #         [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        #     sampling_locations = reference_points[:, :, None, :, None, :] \
        #         + sampling_offsets \
        #         / offset_normalizer[None, None, None, :, None, :]

        # elif reference_points.shape[-1] == 4:
        #     sampling_locations = reference_points[:, :, None, :, None, :2] \
        #         + sampling_offsets / self.num_points \
        #         * reference_points[:, :, None, :, None, 2:] \
        #         * 0.5
        # else:
        #     raise ValueError(
        #         f'Last dim of reference_points must be'
        #         f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        
        output = self.output_proj(deformable_attn_pytorch(value, (h, w), sampling_locations, attention_weights))
        
        # return self.dropout(output) + identity
        return self.dropout(output) + identity, sampling_offsets, attention_weights
        
    
    def get_reference_points(self, H, W, bs=1, device='cuda', dtype=torch.half):
        ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.repeat(bs, 1, 1)
        return ref_2d

class DeformableSpatialAttentionModule(nn.Module):
    def __init__(self, 
                 embed_dims,
                 H,
                 W,
                 n_layers=8,
                 num_heads=8,
                 num_points=12,
                 dropout=0.1,
                 num_frames=3):
        super(DeformableSpatialAttentionModule, self).__init__()
        self.embed_dims = embed_dims
        self.positional_encoding = LearnedPositionalEncoding(embed_dims//2, H, W)
        self.attention_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.attention_layers.append(DeformableSpatialAttentionLayer(embed_dims, num_heads, num_points, dropout))
    
    def forward(self,
                layer,
                query,
                key=None,
                value=None,
                identity=None,
                device='cuda',
                dtype=torch.half,
                spatial_shapes=None):
        
        bs, num_query, embed_dims = query.shape
        h, w = spatial_shapes
        pos_mask = torch.zeros((bs, h, w), device=device).to(dtype)
        query_pos = self.positional_encoding(pos_mask).to(dtype).flatten(2).transpose(1,2) # bs, num_query, embed_dims=pos_dim*2
        
        return self.attention_layers[layer](query=query,
                                            key=key,
                                            value=value,
                                            query_pos=query_pos,
                                            identity=identity,
                                            device=device,
                                            dtype=dtype,
                                            spatial_shapes=spatial_shapes)
        
class DeformableSpatialAttentionModuleWithTemporalFusion(DeformableSpatialAttentionModule):
    def __init__(self, 
                 embed_dims,
                 H,
                 W,
                 n_layers=8,
                 num_heads=8,
                 num_points=12,
                 dropout=0.1,
                 num_frames=3):
        super().__init__(embed_dims=embed_dims,
                         H=H,
                         W=W,
                         n_layers=n_layers,
                         num_heads=num_heads,
                         num_points=num_points,
                         dropout=dropout)
        self.num_frames = num_frames
        self.positional_encoding = LearnedPositionalEncoding(embed_dims//2, H, W)
        self.attention_layers = nn.ModuleList()
        self.temporal_fus_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.temporal_fus_layers.append(TAdaConv2dV2(embed_dims, embed_dims, [1,3,3]))
        # self.temporal_fus = TAdaConv2dV2(embed_dims, embed_dims, [1,3,3])
        for _ in range(n_layers):
            self.attention_layers.append(DeformableSpatialAttentionLayer(embed_dims, num_heads, num_points, dropout))
    
    def forward(self,
                layer,
                query,
                key=None,
                value=None,
                identity=None,
                device='cuda',
                dtype=torch.half,
                spatial_shapes=None):
        feat_out, normalized_sampling_offsets, attention_weights = super().forward(layer=layer,
                        query=query,
                        key=key,
                        value=value,
                        identity=identity,
                        device=device,
                        dtype=dtype,
                        spatial_shapes=spatial_shapes)
        bs, num_query, embed_dims = query.shape
        num_frames = self.num_frames
        bs_org = bs // num_frames
        h, w = spatial_shapes
        
        # bs, num_query, embed_dims -> bs_org, channels, num_frames, h,w
        feat_out = feat_out.transpose(1,2).reshape(bs_org, num_frames, embed_dims, h, w).transpose(1,2)
        feat_out += self.temporal_fus_layers[layer](feat_out)
        # feat_out += self.temporal_fus(feat_out)
        # bs_org, channels, num_frames, h,w -> bs, num_query, embed_dims
        feat_out = feat_out.transpose(1,2).reshape(bs, embed_dims, h*w).transpose(1, 2)
        return feat_out, normalized_sampling_offsets, attention_weights
    

class DeformableSpatialAttentionModuleWithTemporalFusionSharedConv(DeformableSpatialAttentionModule):
    def __init__(self, 
                 embed_dims,
                 H,
                 W,
                 n_layers=8,
                 num_heads=8,
                 num_points=12,
                 dropout=0.1,
                 num_frames=3):
        super().__init__(embed_dims=embed_dims,
                         H=H,
                         W=W,
                         n_layers=n_layers,
                         num_heads=num_heads,
                         num_points=num_points,
                         dropout=dropout)
        self.num_frames = num_frames
        self.positional_encoding = LearnedPositionalEncoding(embed_dims//2, H, W)
        self.attention_layers = nn.ModuleList()
        self.temporal_fus_layers = nn.ModuleList()
        self.temporal_fus = TAdaConv2dV2(embed_dims, embed_dims, [1,3,3])
        for _ in range(n_layers):
            self.attention_layers.append(DeformableSpatialAttentionLayer(embed_dims, num_heads, num_points, dropout))
    
    def forward(self,
                layer,
                query,
                key=None,
                value=None,
                identity=None,
                device='cuda',
                dtype=torch.half,
                spatial_shapes=None):
        feat_out, normalized_sampling_offsets, attention_weights = super().forward(layer=layer,
                        query=query,
                        key=key,
                        value=value,
                        identity=identity,
                        device=device,
                        dtype=dtype,
                        spatial_shapes=spatial_shapes)
        bs, num_query, embed_dims = query.shape
        num_frames = self.num_frames
        bs_org = bs // num_frames
        h, w = spatial_shapes
        
        # bs, num_query, embed_dims -> bs_org, channels, num_frames, h,w
        feat_out = feat_out.transpose(1,2).reshape(bs_org, num_frames, embed_dims, h, w).transpose(1,2)
        feat_out += self.temporal_fus(feat_out)
        # bs_org, channels, num_frames, h,w -> bs, num_query, embed_dims
        feat_out = feat_out.transpose(1,2).reshape(bs, embed_dims, h*w).transpose(1, 2)
        return feat_out, normalized_sampling_offsets, attention_weights

class DeformableSpatialAttention(nn.Module):
    """Deformable Attention Module."""
    
    def __init__(self, 
                 embed_dims,
                 H,
                 W,
                 n_layers=8,
                 num_heads=8,
                 num_points=12,
                 dropout=0.1,
                 num_frames=3):
        super(DeformableSpatialAttention, self).__init__()
        self.n_layers = n_layers
        self.embed_dims = embed_dims
        self.H = H
        self.W = W
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.num_points = num_points
        self.dropout = dropout
        self.num_frames = num_frames
        self.temporal_fus_rgb = None
        self.temporal_fus_ir = None
        self.initModule()
        
    def initModule(self):
        self.rgb_attention = DeformableSpatialAttentionModule(self.embed_dims, self.H, self.W, self.n_layers, self.num_heads, self.num_points, self.dropout)
        self.ir_attention = DeformableSpatialAttentionModule(self.embed_dims, self.H, self.W, self.n_layers, self.num_heads, self.num_points, self.dropout)
        
        
    
    def forward(self, x):
        '''
        Args:
            x (tuple)
        return:
        '''
        rgb_fea = x[0]  # rgb_fea (tensor): dim:(B, C, num_frames, H, W)
        ir_fea = x[1]   # ir_fea (tensor): dim:(B, C, num_frames, H, W)
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs_org, c_org, num_frames, h, w = rgb_fea.shape
        rgb_fea = rgb_fea.transpose(1,2).reshape(bs_org*num_frames, c_org, h,w)
        ir_fea = ir_fea.transpose(1,2).reshape(bs_org*num_frames, c_org, h,w)
        bs, embed_dims, h, w = rgb_fea.shape
        
        rgb_fea = rgb_fea.reshape(bs, embed_dims, h*w).transpose(1, 2)  # bs, h*w, embed_dims
        ir_fea = ir_fea.reshape(bs, embed_dims, h*w).transpose(1, 2)  # bs, h*w, embed_dims
    
        normalized_sampling_offsets_list = []
        attention_weights_list = []
        for layer in range(self.n_layers):
            try:
                rgb_fea_out, normalized_sampling_offsets, attention_weights = self.rgb_attention(layer=layer,
                                                query=rgb_fea, 
                                                key=ir_fea, 
                                                value=ir_fea, 
                                                identity=rgb_fea, 
                                                device=rgb_fea.device, 
                                                dtype=rgb_fea.dtype,
                                                spatial_shapes=(h, w))
                
                
            except Exception as e:
                print('rgb_fea')
                import traceback; traceback.print_exc()
                import pdb; pdb.set_trace()
                
            try:
                ir_fea_out, normalized_sampling_offsets, attention_weights = self.ir_attention(layer=layer,
                                                query=ir_fea, 
                                                key=rgb_fea, 
                                                value=rgb_fea, 
                                                identity=ir_fea,
                                                device=ir_fea.device,
                                                dtype=ir_fea.dtype,
                                                spatial_shapes=(h, w))
                # normalized_sampling_offsets_list.append(normalized_sampling_offsets)
                # normalized_sampling_offsets_list.append(normalized_sampling_offsets)
                # attention_weights_list.append(attention_weights)
            
            except Exception as e:
                print('ir_fea')
                import traceback; traceback.print_exc()
                import pdb; pdb.set_trace()

            
            rgb_fea = rgb_fea_out
            ir_fea = ir_fea_out
            # rgb_fea = rgb_fea_out
        # import pdb; pdb.set_trace()
        # torch.Size([64, 3, 80, 80, 8, 12, 2])
        
        
        
        if self.temporal_fus_rgb is not None:
            # bs, num_query, embed_dims -> bs_org, channels, num_frames, h,w
            rgb_fea = rgb_fea.transpose(1,2).reshape(bs_org, num_frames, embed_dims, h, w).transpose(1,2)
            rgb_fea += self.temporal_fus_rgb(rgb_fea)
            # bs_org, channels, num_frames, h,w -> bs, num_query, embed_dims
            rgb_fea = rgb_fea.transpose(1,2).reshape(bs, embed_dims, h*w).transpose(1, 2)
        
        if self.temporal_fus_ir is not None:
            # bs, num_query, embed_dims -> bs_org, channels, num_frames, h,w
            ir_fea = ir_fea.transpose(1,2).reshape(bs_org, num_frames, embed_dims, h, w).transpose(1,2)
            ir_fea += self.temporal_fus_ir(ir_fea)
            # bs_org, channels, num_frames, h,w -> bs, num_query, embed_dims
            ir_fea = ir_fea.transpose(1,2).reshape(bs, embed_dims, h*w).transpose(1, 2)
        

       
        
        #######################################################################
        # Attention Visualization
        #######################################################################
        if False:
            try: 
                from tqdm import tqdm
                origin = torch.Tensor([[30, 30]])
                weight = torch.Tensor([[1]])
                for layer in tqdm(range(self.n_layers)[:1]):
                    origin_new = []
                    weight_new = []
                    normalized_sampling_offsets = normalized_sampling_offsets_list[layer].detach().cpu()
                    normalized_sampling_offsets_reshaped = normalized_sampling_offsets.reshape(bs_org, num_frames, h, w, 8 ,12 ,2)
                    attention_weights = attention_weights_list[layer].detach().cpu()
                    attention_weights_reshaped = attention_weights.reshape(bs_org, num_frames, h, w, 8 ,12)
                    
                    origin = origin.to(torch.int64)
                    for idx, sample_point in tqdm(enumerate(origin)):
                        if sample_point[0] < 0 or sample_point[0] >= h or sample_point[1] < 0 or sample_point[1] >= w:
                            continue
                        offsets = normalized_sampling_offsets_reshaped[0,0,sample_point[0], sample_point[1]]
                        origin_new.append(sample_point[None, None] + offsets)
                        weights = attention_weights_reshaped[0,0,sample_point[0], sample_point[1]] / 8
                        weight_new.append(weights * weight[idx])
                        
                    origin = torch.cat(origin_new, dim=0).reshape(-1, 2).to(torch.int64)
                    weight = torch.cat(weight_new, dim=0).reshape(-1, 1).to(torch.float32)
                    # 获取每个唯一 origin 的索引和出现次数
                    unique_origins, inverse_indices, counts = torch.unique(origin, dim=0, return_inverse=True, return_counts=True)

                    # 初始化一个新的 weight 数组，用于存储累加的值
                    accumulated_weight = torch.zeros(unique_origins.size(0), 1, dtype=torch.float32)

                    # 对于每个唯一的 origin，累加其对应的 weight
                    for i in range(unique_origins.size(0)):
                        # 选取具有相同 origin 对的所有 weight 并累加
                        accumulated_weight[i] = weight[inverse_indices == i].sum()

                    # 更新 origin 为唯一值
                    origin = unique_origins

                    # 将累加后的 weight 赋值给 weight 变量
                    weight = accumulated_weight
                    
                    print(origin.shape)
            except:
                import traceback; traceback.print_exc()
                import pdb; pdb.set_trace()
            print("pass")
            filtered_tensor = origin[(origin[:, 0] >= 0) & (origin[:, 1] >= 0) & (origin[:, 0] < 80) & (origin[:, 1] < 80)]
            filtered_weight = weight[(origin[:, 0] >= 0) & (origin[:, 1] >= 0) & (origin[:, 0] < 80) & (origin[:, 1] < 80)]
            points = filtered_tensor.numpy()

            from PIL import Image
            # Creating a blank white image
            # image_size = filtered_tensor.max().item() + 1
            image_size = 80
            img = Image.new('RGB', (image_size, image_size), 'white')
            pixels = img.load()

            filtered_weight_abs = filtered_weight.abs()
            max_weight = filtered_weight_abs.max().item()
            min_weight = filtered_weight_abs.min().item()
            filtered_weight_abs = (filtered_weight_abs - min_weight) / (max_weight - min_weight)
            # Drawing the points on the image
            for idx, point in enumerate(points):
                # Scaling the points to fit in the image dimensions
                x = int(point[0])
                y = int(point[1])
                if x < image_size and y < image_size:
                    if filtered_weight_abs[idx] > 0.3:
                        pixel_value = 255 - int(filtered_weight_abs[idx] * 255)
                        pixels[x, y] = (pixel_value, pixel_value, pixel_value)  # Black color
                    # pixels[x, y] = (0, 0, 0)  # Black color
            pixels[30,30] = (255, 0, 0)
                    
            img.save('logs/temp/deformable_visual_ir_rgb_ir_layer0.png')        
            
            import pdb; pdb.set_trace()
            
        
                
                
                
                
                    
        # bs, num_query, embed_dims -> bs_org, channels, num_frames, h,w
        rgb_fea = rgb_fea.transpose(1,2).reshape(bs_org, num_frames, embed_dims, h,w).transpose(1,2)
        ir_fea = ir_fea.transpose(1,2).reshape(bs_org, num_frames, embed_dims, h,w).transpose(1,2)
        return rgb_fea, ir_fea
            
            


class DeformableSpatialAttentionWithTemporalFusionV1(DeformableSpatialAttention):
    """Deformable Attention Module."""
    
    def initModule(self):
        self.rgb_attention = DeformableSpatialAttentionModule(self.embed_dims, self.H, self.W, self.n_layers, self.num_heads, self.num_points, self.dropout, self.num_frames)
        self.ir_attention = DeformableSpatialAttentionModule(self.embed_dims, self.H, self.W, self.n_layers, self.num_heads, self.num_points, self.dropout, self.num_frames)
        self.temporal_fus_rgb = TAdaConv2dV2(self.embed_dims, self.embed_dims, [1,3,3])
        self.temporal_fus_ir = TAdaConv2dV2(self.embed_dims, self.embed_dims, [1,3,3])
        
class DeformableSpatialAttentionWithTemporalFusionV2(DeformableSpatialAttention):
    """Deformable Attention Module."""
    
    def initModule(self):
        self.rgb_attention = DeformableSpatialAttentionModuleWithTemporalFusionSharedConv(self.embed_dims, self.H, self.W, self.n_layers, self.num_heads, self.num_points, self.dropout, self.num_frames)
        self.ir_attention = DeformableSpatialAttentionModuleWithTemporalFusionSharedConv(self.embed_dims, self.H, self.W, self.n_layers, self.num_heads, self.num_points, self.dropout, self.num_frames)

class DeformableSpatialAttentionWithTemporalFusionV3(DeformableSpatialAttention):
    """Deformable Attention Module."""
    
    def initModule(self):
        self.rgb_attention = DeformableSpatialAttentionModuleWithTemporalFusion(self.embed_dims, self.H, self.W, self.n_layers, self.num_heads, self.num_points, self.dropout, self.num_frames)
        self.ir_attention = DeformableSpatialAttentionModuleWithTemporalFusion(self.embed_dims, self.H, self.W, self.n_layers, self.num_heads, self.num_points, self.dropout, self.num_frames)


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, d_model, h=8, block_exp=4,
                 n_layer=8, vert_anchors=8, horz_anchors=8,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * vert_anchors * horz_anchors, self.n_embd))

        # transformer
        self.trans_blocks = nn.Sequential(*[model_modules_shared.myTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)
                                            for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)

        # regularization
        self.drop = nn.Dropout(embd_pdrop)

        # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))
        # self.avgpool = nn.AdaptiveMaxPool2d((self.vert_anchors, self.horz_anchors))

        # init weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
        Args:
            x (tuple?)

        """
        rgb_fea = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
        ir_fea = x[1]   # ir_fea (tensor): dim:(B, C, H, W)
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        if len(rgb_fea.shape) == 5:
            bs, c, num_frames, h, w = rgb_fea.shape
            rgb_fea = rgb_fea.reshape(bs,c*num_frames, h,w)    
            ir_fea = ir_fea.reshape(bs,c*num_frames, h,w)    
        bs, c, h, w = rgb_fea.shape

        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation
        rgb_fea = self.avgpool(rgb_fea)
        ir_fea = self.avgpool(ir_fea)

        # -------------------------------------------------------------------------
        # Transformer
        # -------------------------------------------------------------------------
        # pad token embeddings along number of tokens dimension
        rgb_fea_flat = rgb_fea.view(bs, c, -1)  # flatten the feature
        ir_fea_flat = ir_fea.view(bs, c, -1)  # flatten the feature
        token_embeddings = torch.cat([rgb_fea_flat, ir_fea_flat], dim=2)  # concat
        token_embeddings = token_embeddings.permute(0, 2, 1).contiguous()  # dim:(B, 2*H*W, C)

        # transformer
        x = self.drop(self.pos_emb + token_embeddings)  # sum positional embedding and token    dim:(B, 2*H*W, C)
        x = self.trans_blocks(x)  # dim:(B, 2*H*W, C)

        # decoder head
        x = self.ln_f(x)  # dim:(B, 2*H*W, C)
        x = x.view(bs, 2, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)

        # 这样截取的方式, 是否采用映射的方式更加合理？
        rgb_fea_out = x[:, 0, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        ir_fea_out = x[:, 1, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)

        # -------------------------------------------------------------------------
        # Interpolate (or Upsample)
        # -------------------------------------------------------------------------
        rgb_fea_out = F.interpolate(rgb_fea_out, size=([h, w]), mode='bilinear')
        ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode='bilinear')

        return rgb_fea_out, ir_fea_out
    

class TadaConvSpatialGPT(GPT):
    def forward(self, x):
        """
        Args:
            x (tuple?)

        """
        rgb_fea = x[0]  # rgb_fea (tensor): dim:(B, C, num_frames, H, W)
        ir_fea = x[1]   # ir_fea (tensor): dim:(B, C, num_frames, H, W)
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs_org, c_org, num_frames, h, w = rgb_fea.shape
        rgb_fea = rgb_fea.transpose(1,2).reshape(bs_org*num_frames, c_org, h,w)
        ir_fea = ir_fea.transpose(1,2).reshape(bs_org*num_frames, c_org, h,w) 
        bs, c, h, w = rgb_fea.shape

        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation
        rgb_fea = self.avgpool(rgb_fea)
        ir_fea = self.avgpool(ir_fea)

        # -------------------------------------------------------------------------
        # Transformer
        # -------------------------------------------------------------------------
        # pad token embeddings along number of tokens dimension
        rgb_fea_flat = rgb_fea.view(bs, c, -1)  # flatten the feature
        ir_fea_flat = ir_fea.view(bs, c, -1)  # flatten the feature
        token_embeddings = torch.cat([rgb_fea_flat, ir_fea_flat], dim=2)  # concat
        token_embeddings = token_embeddings.permute(0, 2, 1).contiguous()  # dim:(B, 2*H*W, C)

        # transformer
        x = self.drop(self.pos_emb + token_embeddings)  # sum positional embedding and token    dim:(B, 2*H*W, C)
        x = self.trans_blocks(x)  # dim:(B, 2*H*W, C)

        # decoder head
        x = self.ln_f(x)  # dim:(B, 2*H*W, C)
        x = x.view(bs, 2, self.vert_anchors, self.horz_anchors, self.n_embd)
        self.x = x.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)

        # 这样截取的方式, 是否采用映射的方式更加合理？ ( In this way of interception, is it more reasonable to use mapping?)
        rgb_fea_out = self.x[:, 0, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        ir_fea_out = self.x[:, 1, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)

        # -------------------------------------------------------------------------
        # Interpolate (or Upsample)
        # -------------------------------------------------------------------------
        rgb_fea_out = F.interpolate(rgb_fea_out, size=([h, w]), mode='bilinear')
        ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode='bilinear')

        rgb_fea_out = rgb_fea_out.reshape(bs_org, num_frames, c_org, h, w).transpose(1,2) # back to bs, channels, num_frames, h,w
        ir_fea_out = ir_fea_out.reshape(bs_org, num_frames, c_org, h, w).transpose(1,2) # back to bs, channels, num_frames, h,w

        
        return rgb_fea_out, ir_fea_out


    # def __init__(self, d_model, h=8, block_exp=4,
    #              n_layer=8, vert_anchors=8, horz_anchors=8,
    #              embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):

class TadaConvSpatialTemporalGPT(TadaConvSpatialGPT):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, d_model, h=8, block_exp=4,
                 n_layer=8, vert_anchors=8, horz_anchors=8):
        
        super().__init__(d_model=d_model)
        self.temporal_fus = TAdaConv2dV2(2*d_model, d_model, [1,3,3])

    def forward(self, x):
        """
        Args:
            x (tuple?)

        """
        # import pdb; pdb.set_trace()
        rgb_fea_out, ir_fea_out =  super().forward(x) # Spatial fusion
        
        bs_org, c_org, num_frames, h, w = rgb_fea_out.shape
        ###########################################################################################################        
        temp_feat = self.x.reshape(bs_org, num_frames, 2, c_org, self.vert_anchors, self.horz_anchors) 
        temp_feat = temp_feat.permute(0,2,3,1,4,5).reshape(bs_org, 2*c_org, num_frames, self.vert_anchors, self.horz_anchors)
        temp_fusion = self.temporal_fus(temp_feat) # temporal fusion
        
        # -------------------------------------------------------------------------
        # Interpolate (or Upsample)
        # -------------------------------------------------------------------------
        temp_feat_out = F.interpolate(temp_fusion, [num_frames, h, w], mode = 'trilinear')
        
        rgb_fea_out = rgb_fea_out + temp_feat_out   # spatial + temporal fusion,  (bs, channels, num_frames, h,w)
        ir_fea_out = ir_fea_out + temp_feat_out     # spatial + temporal fusion,  (bs, channels, num_frames, h,w)
        

        return rgb_fea_out, ir_fea_out
    

class TadaConvSpatialTemporalGPTv2(TadaConvSpatialGPT):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, d_model, h=8, block_exp=4,
                 n_layer=8, vert_anchors=8, horz_anchors=8):
        
        super().__init__(d_model=d_model)
        self.pos_emb  = nn.Parameter(torch.zeros(1, 2 * vert_anchors * horz_anchors * 3, d_model))

    def forward(self, x):
        """
        Args:
            x (tuple?)

        """
        rgb_fea = x[0]  # rgb_fea (tensor): dim:(B, C, num_frames, H, W)
        ir_fea = x[1]   # ir_fea (tensor): dim:(B, C, num_frames, H, W)
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs_org, c_org, num_frames, h, w = rgb_fea.shape
        rgb_fea = rgb_fea.transpose(1,2).reshape(bs_org*num_frames, c_org, h,w)
        ir_fea = ir_fea.transpose(1,2).reshape(bs_org*num_frames, c_org, h,w) 
        # bs, c, h, w = rgb_fea.shape

        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation
        rgb_fea = self.avgpool(rgb_fea)
        ir_fea = self.avgpool(ir_fea)
        rgb_fea = rgb_fea.reshape(bs_org, num_frames, c_org, self.vert_anchors, self.horz_anchors).transpose(1,2)
        ir_fea = ir_fea.reshape(bs_org, num_frames, c_org, self.vert_anchors, self.horz_anchors).transpose(1,2)

        # -------------------------------------------------------------------------
        # Transformer
        # -------------------------------------------------------------------------
        # pad token embeddings along number of tokens dimension
        rgb_fea_flat = rgb_fea.reshape(bs_org, c_org, -1)  # flatten the feature
        ir_fea_flat = ir_fea.reshape(bs_org, c_org, -1)  # flatten the feature
        token_embeddings = torch.cat([rgb_fea_flat, ir_fea_flat], dim=2)  # concat
        token_embeddings = token_embeddings.permute(0, 2, 1).contiguous()  # dim:(B, 2*H*W, C)

        # transformer
        x = self.drop(self.pos_emb + token_embeddings)  # sum positional embedding and token    dim:(B, 2*H*W, C)
        x = self.trans_blocks(x)  # dim:(B, 2*H*W, C)

        # decoder head
        x = self.ln_f(x)  # dim:(B, 2*H*W, C)
        x = x.view(bs_org, 2, num_frames, self.vert_anchors, self.horz_anchors, self.n_embd)
        self.x = x.permute(0, 1, 2, 5, 3, 4)  # dim:(B, 2, C, H, W)

        # 这样截取的方式, 是否采用映射的方式更加合理？ ( In this way of interception, is it more reasonable to use mapping?)
        rgb_fea_out = self.x[:, 0, :, :, :, :].contiguous().view(bs_org * num_frames, self.n_embd, self.vert_anchors, self.horz_anchors)
        ir_fea_out = self.x[:, 1, :, :, :, :].contiguous().view(bs_org * num_frames, self.n_embd, self.vert_anchors, self.horz_anchors)

        # -------------------------------------------------------------------------
        # Interpolate (or Upsample)
        # -------------------------------------------------------------------------
        rgb_fea_out = F.interpolate(rgb_fea_out, size=([h, w]), mode='bilinear')
        ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode='bilinear')

        rgb_fea_out = rgb_fea_out.reshape(bs_org, num_frames, c_org, h, w).transpose(1,2) # back to bs, channels, num_frames, h,w
        ir_fea_out = ir_fea_out.reshape(bs_org, num_frames, c_org, h, w).transpose(1,2) # back to bs, channels, num_frames, h,w

        
        # return rgb_fea_out, ir_fea_out

        return rgb_fea_out, ir_fea_out


# class MLPSpatialTemporalMixer(nn.Module):
#     def __init__(self, channels_mlp_dim, rgb_ir_token_mlp_dim, patches, num_blocks, num_frames):
#         super().__init__()
#         self.patches = patches
#         self.num_frames = num_frames
#         self.num_blocks = num_blocks
#         self.token_mlp_dim = rgb_ir_token_mlp_dim
#         self.channels_mlp_dim = channels_mlp_dim
#         self.temporal_mlp_dim = num_frames*patches
#         self.mixerblock = nn.Sequential(*[MixerBlock(rgb_ir_token_mlp_dim, channels_mlp_dim, num_frames, self.temporal_mlp_dim) for _ in range(num_blocks)])

#         # self.mixer1 = MixerBlock(tokens_mlp_dim, channels_mlp_dim, temporal_mlp_dim)
    
#     def forward(self, inputs):
#         rgb_feat = inputs[0]
#         ir_feat = inputs[1]
#         b, c, f, h, w = rgb_feat.shape

#         patch = int(math.sqrt(self.token_mlp_dim/2))
#         initial_patches_rgb = rearrange(rgb_feat, 'b c f (h ph) (w pw) -> b f (h w) (ph pw) c', ph = patch , pw = patch)
#         initial_patches_ir = rearrange(ir_feat, 'b c f (h ph) (w pw) -> b f (h w) (ph pw) c', ph = patch , pw = patch)
#         initial_patches = torch.cat([initial_patches_rgb, initial_patches_ir], dim=3)  # concat
        
#         output_patches = self.mixerblock(initial_patches) # Inital patches and output patches have same dim and and patched similarly
        
#         y = rearrange(output_patches,' b f p s c-> b c f p s')
#         dim = int(math.sqrt(y.shape[-2]))
#         rgb_feat_fus = y[:,:,:,:,:self.token_mlp_dim//2]
#         ir_feat_fus = y[:,:,:,:,self.token_mlp_dim//2:]
#         rgb_feat_fus = rearrange(rgb_feat_fus, 'b c f (p1 p2) (ph pw)->b c f (p1 ph) (p2 pw)', p1=dim,p2=dim, ph=patch, pw=patch)
#         ir_feat_fus = rearrange(ir_feat_fus, 'b c f (p1 p2) (ph pw)->b c f (p1 ph) (p2 pw)', p1=dim,p2=dim, ph=patch, pw=patch)


#         return rgb_feat_fus, ir_feat_fus

# class MixerBlock(nn.Module):
#     def __init__(self, tokens_mlp_dim, channels_mlp_dim, num_of_frames, temporal_mlp_dim):
#         super().__init__()
#         self.rgb_ir_spatial_mix = MlpBlock(tokens_mlp_dim) 
#         self.channel_mixing = MlpBlock(channels_mlp_dim)
#         self.temporal_mix = MlpBlock(num_of_frames)
#         # might make sense to either use larger tokens so that temporal relations are easier captured
#         # could image that if we have a small window it would be hard to find 
#         self.spatial_temporal_patch_mix = MlpBlock(temporal_mlp_dim)
#         self.norm1  = nn.LayerNorm(channels_mlp_dim) #???
#         self.norm2  = nn.LayerNorm(channels_mlp_dim) #???
#         self.norm3  = nn.LayerNorm(channels_mlp_dim) #???
#         self.norm4  = nn.LayerNorm(channels_mlp_dim) #???

        
#     def forward(self, x):
#         b, f ,p, s, c = x.shape #batch, frame, patch_num, token_mlp_dim, channels

#         y = self.norm1(x)
#         y = rearrange(y, 'b f p s c -> b f p c s')
#         y = self.rgb_ir_spatial_mix(y)
#         y = rearrange(y, 'b f p c s-> b f p s c')
#         x = x + y
        
#         y = self.norm2(x)
#         y = self.channel_mixing(y)
#         x = x + y    
                
#         y = self.norm3(x)
#         y = rearrange(y, 'b f p s c -> b p s c f')
#         y = self.temporal_mix(y)
#         y = rearrange(y, 'b p s c f -> b f p s c')
#         x = x + y
        
#         y = self.norm4(x)
#         y = rearrange(y, 'b f p s c -> b c s (f p)')
#         y = self.spatial_temporal_patch_mix(y)
#         y = rearrange(y,' b c s (f p) -> b f p s c', f = f)
#         x = x + y
        
#         return x
    
# class MlpBlock(nn.Module):
#     def __init__(self, mlp_dim):
#         super().__init__()
#         self.mlp_dim = mlp_dim
#         self.mlp1 = nn.Linear(mlp_dim, mlp_dim)
#         self.gelu = nn.GELU()
#         self.mlp2 = nn.Linear(mlp_dim, mlp_dim)
    
#     def forward(self, x):
#         y = self.mlp1(x)
#         y = self.gelu(y)
#         return self.mlp2(y)
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



# class MLPTemporalMixerv2(nn.Module):
#     def __init__(self, channels_mlp_dim, rgb_ir_tokens_mlp_dim,
#                 num_temporal_mix_block=2, num_patches=16, num_of_frames=3, pdrop=0.1):
#         super().__init__()
#         self.rgb_ir_tokens_mlp_dim = rgb_ir_tokens_mlp_dim
#         self.mixerblock = TemporalBlockv2(channels_mlp_dim, num_temporal_mix_block, num_patches,num_of_frames, pdrop)

#     def before_mixer(self, inputs):
#         rgb_feat = inputs[0]
#         ir_feat = inputs[1]
#         b, c, f, h, w = rgb_feat.shape
        
#         patch = int(math.sqrt(self.rgb_ir_tokens_mlp_dim/2))
#         initial_patches_rgb = rearrange(rgb_feat, 'b c f (h ph) (w pw) -> b f (h w) (ph pw) c', ph = patch , pw = patch)
#         initial_patches_ir = rearrange(ir_feat, 'b c f (h ph) (w pw) -> b f (h w) (ph pw) c', ph = patch , pw = patch)
#         initial_patches = torch.cat([initial_patches_rgb, initial_patches_ir], dim=3)  # concat
#         return initial_patches, patch
    
#     def after_mixer(self, output_patches, patch):
#         y = rearrange(output_patches,' b f p s c-> b c f p s')
#         dim = int(math.sqrt(y.shape[-2]))
#         rgb_feat_fus = y[:,:,:,:,:self.rgb_ir_tokens_mlp_dim//2]
#         ir_feat_fus = y[:,:,:,:,self.rgb_ir_tokens_mlp_dim//2:]
#         rgb_feat_fus = rearrange(rgb_feat_fus, 'b c f (p1 p2) (ph pw)->b c f (p1 ph) (p2 pw)', p1=dim,p2=dim, ph=patch, pw=patch)
#         ir_feat_fus = rearrange(ir_feat_fus, 'b c f (p1 p2) (ph pw)->b c f (p1 ph) (p2 pw)', p1=dim,p2=dim, ph=patch, pw=patch)

#         return rgb_feat_fus, ir_feat_fus
    
#     def forward(self, inputs):
#         initial_patches, patch = self.before_mixer(inputs)
#         output_patches = self.mixerblock(initial_patches)
        
#         rgb_feat_fus, ir_feat_fus = self.after_mixer(output_patches, patch)

#         return rgb_feat_fus, ir_feat_fus    
