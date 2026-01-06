import torch
import torch.nn as nn
from  torchvision.ops  import deform_conv2d
from models.spatiotemporal_sampling_network import SpatialTemporalSampling
from models.spatiotemporal_sampling_network_simpler import SpatialTemporalSamplingv2
from datetime import datetime
from utils.post_process_fusion import fuse_predictions
class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        # import pdb; pdb.set_trace()
        feature = x.copy()
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            print(x[i].dtype)
            print(x[i].device)
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device )

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x, feature)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

class LastFrameDeformDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(LastFrameDeformDetect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(SpatialTemporalSampling(x, self.no*self.na, 3,3) for x in ch) #output deform conv

    def forward(self, x):
        # import pdb; pdb.set_trace()
        feature = x.copy()
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            # if x[i].isnan().any():
            #     raise RuntimeError("ERROR: Got NaN in {}".format(datetime.now()))
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device )

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x, feature)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


# class LastFrameDeformDetectv2(LastFrameDeformDetect):

#     def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
#         super(LastFrameDeformDetectv2, self).__init__()
#         self.nc = nc  # number of classes
#         self.no = nc + 5  # number of outputs per anchor
#         self.nl = len(anchors)  # number of detection layers
#         self.na = len(anchors[0]) // 2  # number of anchors
#         self.grid = [torch.zeros(1)] * self.nl  # init grid
#         a = torch.tensor(anchors).float().view(self.nl, -1, 2)
#         self.register_buffer('anchors', a)  # shape(nl,na,2)
#         self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
#         self.m = nn.ModuleList(SpatialTemporalSamplingv2(x, self.no*self.na, 3,3) for x in ch) #output deform conv


class LastFrameDetect(Detect):

    def forward(self, x):
        
        feature = x.copy()
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i].permute(2,0,1,3,4)[0])  # conv, (using [0] because in dataloader getitem used '::-1' and then reshaped)
            # x[i] = self.m[i](x[i].mean(dim=2))  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device )
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x, feature)
    
    
    
class ThermalRgbDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors) # number of detection layers is doubled because thermal and RGB
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl*2  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.conv_modules_init(ch)
        self.fuseconv_init(ch)

    def conv_modules_init(self, ch):
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv for RGB
        self.m_ir = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv for Thermal

    def fuseconv_init(self, ch):
        self.conv = nn.ModuleList(nn.Conv3d(in_channels=dim*2, out_channels=dim ,kernel_size=1, stride=1, padding=0) for dim in ch)
        self.conv_ir = nn.ModuleList(nn.Conv3d(in_channels=dim*2, out_channels=dim ,kernel_size=1, stride=1, padding=0) for dim in ch)
    
    def fuseconv(self, x):
        fused_bb_rgb_thermal = x[:3]
        fused_in_head =  x[3:]
        out = []
        out_ir = []
        for i, xx in enumerate(zip(fused_bb_rgb_thermal)):
            out.append(self.conv[i](torch.cat((xx[0][0], fused_in_head[i]),dim=1))) # RGB
            out_ir.append(self.conv_ir[i](torch.cat((xx[0][1], fused_in_head[i]),dim=1))) # Thermal
        out.extend(out_ir)
        return out
    
    def forward(self, x):
        feature = x.copy()
        # x = x.copy()  # for profiling
        x = self.fuseconv(x)
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl*2):
            j = i % 3 # first 3 are RGB, last 3 are thermal
            if i < 3:
                x[i] = self.m[j](x[i])  # conv
            else:
                x[i] = self.m_ir[j](x[i])  # conv

            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device )

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[j]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[j]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x, feature)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

class LastFrameThermalRgbDetect(ThermalRgbDetect):
    def fuseconv_init(self, ch):
        self.conv = nn.ModuleList(nn.Sequential(nn.Conv3d(in_channels=dim*2, out_channels=dim ,kernel_size=1, stride=1, padding=0), nn.LeakyReLU()) for dim in ch)
        self.conv_ir = nn.ModuleList(nn.Sequential(nn.Conv3d(in_channels=dim*2, out_channels=dim ,kernel_size=1, stride=1, padding=0), nn.LeakyReLU() ) for dim in ch)
        
    def forward(self, x):
        # import pdb; pdb.set_trace()

        feature = x.copy()
        # x = x.copy()  # for profiling
        x = self.fuseconv(x)
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl*2):
            j = i % 3 # first 3 are RGB, last 3 are thermal
            if i < 3:
                x[i] = self.m[j](x[i].permute(2,0,1,3,4)[0])  # conv, (using [0] because in dataloader getitem used '::-1' and then reshaped)
            else:
                x[i] = self.m_ir[j](x[i].permute(2,0,1,3,4)[0])  # conv, (using [0] because in dataloader getitem used '::-1' and then reshaped)
                
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device )

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[j]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[j]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x, feature)

class LastFrameThermalRgbDetectKLDiv(LastFrameThermalRgbDetect):
    def fuseconv(self, x):
        fused_bb_rgb_thermal = x[:3]
        fused_in_head =  x[3:]
        out = []
        out_ir = []
        for i, xx in enumerate(zip(fused_bb_rgb_thermal)):
            out.append(self.conv[i](torch.cat((xx[0][0], fused_in_head[i]),dim=1))) # RGB
            out_ir.append(self.conv_ir[i](torch.cat((xx[0][1], fused_in_head[i]),dim=1))) # Thermal
        out.extend(out_ir)
        out.append(fused_bb_rgb_thermal)
        return out

class LastFrameThermalRgbDetectKLDivBack(LastFrameThermalRgbDetect):

    def fuseconv(self, x):
        fused_bb_rgb_thermal = x[:3]
        fused_in_head =  x[3:]
        out = []
        out_ir = []
        out_tuple = []
        for i, xx in enumerate(zip(fused_bb_rgb_thermal)):
            out.append(self.conv[i](torch.cat((xx[0][0], fused_in_head[i]),dim=1))) # RGB
            out_ir.append(self.conv_ir[i](torch.cat((xx[0][1], fused_in_head[i]),dim=1))) # Thermal
            out_tuple.append((out[i], out_ir[i]))
        out.extend(out_ir)
        out.append(out_tuple)
        return out
    
class LastFrameThermalRgbDetectKLDivBackNL(LastFrameThermalRgbDetectKLDivBack):

    def fuseconv_init(self, ch):
        self.conv = nn.ModuleList(nn.Sequential(nn.Conv3d(in_channels=dim*2, out_channels=dim ,kernel_size=1, stride=1, padding=0), nn.LeakyReLU()) for dim in ch)
        self.conv_ir = nn.ModuleList(nn.Sequential(nn.Conv3d(in_channels=dim*2, out_channels=dim ,kernel_size=1, stride=1, padding=0), nn.LeakyReLU() ) for dim in ch)

class LastFrameThermalRgbDetectDeform(ThermalRgbDetect):
    def conv_modules_init(self, ch):
        self.m  = nn.ModuleList(SpatialTemporalSampling(x, self.no*self.na, 3,3) for x in ch) #output deform conv
        self.m_ir  = nn.ModuleList(SpatialTemporalSampling(x, self.no*self.na, 3,3) for x in ch) #output deform conv
    
    def fuseconv_init(self, ch):
        self.conv = nn.ModuleList(nn.Sequential(nn.Conv3d(in_channels=dim*2, out_channels=dim ,kernel_size=1, stride=1, padding=0), nn.LeakyReLU()) for dim in ch)
        self.conv_ir = nn.ModuleList(nn.Sequential(nn.Conv3d(in_channels=dim*2, out_channels=dim ,kernel_size=1, stride=1, padding=0), nn.LeakyReLU() ) for dim in ch)
    
    def fuseconv(self, x):
        fused_bb_rgb_thermal = x[:3]
        fused_in_head =  x[3:]
        out = []
        out_ir = []
        out_tuple = []
        for i, xx in enumerate(zip(fused_bb_rgb_thermal)):
            out.append(self.conv[i](torch.cat((xx[0][0], fused_in_head[i]),dim=1))) # RGB
            out_ir.append(self.conv_ir[i](torch.cat((xx[0][1], fused_in_head[i]),dim=1))) # Thermal
            out_tuple.append((out[i], out_ir[i]))
        out.extend(out_ir)
        out.append(out_tuple)
        return out


class ThermalRgbDetectv2(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    def __init__(self, nc=80, anchors=(), ch=()):# frame_used=3, use_tadaconv=False):  # detection layer
        super().__init__()
        # import pdb; pdb.set_trace()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors) # number of detection layers is doubled because thermal and RGB
        # assert self.nl == 6
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.conv_modules_init(ch)

    def conv_modules_init(self, ch):
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv for RGB
        self.m_ir = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv for Thermal
    
    def fuseconv(self, x, scale_offset=3,  use_tadaconv=True, numframes=None):
        # import pdb; pdb.set_trace()
        fused_in_head =  x
        
        out_tuple = []
        for i in  range(3):
            if use_tadaconv:
                out_tuple.append([x[i], x[scale_offset+i]]) # RGB
            else:
                # Reshaping 4D to 5D tensor
                bf, c, h, w = x[i].shape
                bs = bf // numframes
                out_tuple.append([x[i].reshape(bs, numframes, c, h, w).permute(0,2,1,3,4),
                                  x[scale_offset+i].reshape(bs, numframes, c, h, w).permute(0,2,1,3,4)]) # RGB
        
        fused_in_head.append(out_tuple)
        return fused_in_head
    def forward(self, x, use_tadaconv=True, numframes=None):
        feature = x.copy()
        x = self.fuseconv(x, use_tadaconv=use_tadaconv, numframes=numframes)
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl*2):
            j = i % 3 # first 3 are RGB, last 3 are thermal
            
            # Reshaping 4D to 5D tensor
            if not use_tadaconv:
                #b*f,c, h,w => b,c,f,h,w 
                bf, c, h, w = x[i].shape
                bs = bf // numframes
                x[i] = x[i].view(bs, numframes, c, h, w).permute(0,2,1,3,4)
                
            if i < 3:
                x[i] = self.m[j](x[i])  # conv
            else:
                x[i] = self.m_ir[j](x[i])  # conv

            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[j].shape[2:4] != x[i].shape[2:4]:
                    self.grid[j] = self._make_grid(nx, ny).to(x[i].device )

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[j]) * self.stride[j]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[j]  # wh
                z.append(y.view(bs, -1, self.no))


        return x if self.training else (torch.cat(z, 1), x, feature)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

class LastFrameThermalRgbDetectv2(ThermalRgbDetectv2):
    def forward(self, x, use_tadaconv=True, numframes=None, pp_fusion_nms=False):
        feature = x.copy()
        # x = x.copy()  # for profiling
        x = self.fuseconv(x, use_tadaconv=use_tadaconv, numframes=numframes)
        z = []  # inference output
        z_fused = []
        self.training |= self.export
        for i in range(self.nl*2):
            j = i % 3 # first 3 are RGB, last 3 are thermal

            # Reshaping 4D to 5D tensor
            if not use_tadaconv:
                #b*f,c, h,w => b,c,f,h,w 
                bf, c, h, w = x[i].shape
                bs = bf // numframes
                x[i] = x[i].view(bs, numframes, c, h, w).permute(0,2,1,3,4)
            
            if i < 3:
                x[i] = self.m[j](x[i].permute(2,0,1,3,4)[0])  # conv, (using [0] because in dataloader getitem used '::-1' and then reshaped)
            else:
                x[i] = self.m_ir[j](x[i].permute(2,0,1,3,4)[0])  # conv, (using [0] because in dataloader getitem used '::-1' and then reshaped)
                
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[j].shape[2:4] != x[i].shape[2:4]:
                    self.grid[j] = self._make_grid(nx, ny).to(x[i].device )

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[j]) * self.stride[j]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[j]  # wh
                z.append(y.view(bs, -1, self.no))
        
        # Now fuse predictions at each scale (Postprocessing fusion)
        if not self.training and pp_fusion_nms:
            z_rgb = z[:3]
            z_thermal = z[3:]
            for scale_idx in range(3):
                rgb_preds = z_rgb[scale_idx]  # [64, N, 6] where N is 19200, 4800, or 1200
                thermal_preds = z_thermal[scale_idx]  # same shape as rgb_preds
                
                # Fuse predictions at current scale
                fused_scale = fuse_predictions(
                    thermal_preds, 
                    rgb_preds
                )
                z_fused.append(fused_scale)
            return (torch.cat(z_fused, 1), x, feature)

        return x if self.training else (torch.cat(z, 1), x, feature)

class LastFrameThermalRgbDetectDeformv2(ThermalRgbDetectv2):
    def conv_modules_init(self, ch):
        self.m  = nn.ModuleList(SpatialTemporalSamplingv2(x, self.no*self.na, 3,3) for x in ch) #output deform conv
        self.m_ir  = nn.ModuleList(SpatialTemporalSamplingv2(x, self.no*self.na, 3,3) for x in ch) #output deform conv
if __name__ == '__main__':
    # dconv = DeformConv(50,50,3,3)
    # rand = torch.rand(4,50,3,20,20)
    rand = torch.rand(4,1024,3,20,20)
    # ouput = dconv(rand)
    stnet = SpatialTemporalSampling(2*1024, 2*1024, 3,3)
    out = stnet(rand)
    print(out.shape)
    