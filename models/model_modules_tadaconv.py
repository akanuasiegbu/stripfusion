
from .tadaconv_v2 import TAdaConv2dV2 as Conv
import torch.nn as nn
import torch
import models.model_modules_shared as model_modules_shared
import torch.nn.functional as F
class Upsample(nn.Module):
    def __init__(self, size, scale_factor, mode):
        super().__init__()
        self.upsample = nn.Upsample(size, scale_factor, mode)
        self.sf = scale_factor
    def forward(self, x):
        b,c, num_frames, w, h = x.shape
        x = x.reshape(b,-1,w,h)
        return self.upsample(x).reshape(b,c,num_frames, w*self.sf, h*self.sf)

class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, nf, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, (nf) Number of Local + Global Frames in Batch, kernel, stride, padding, groups
        super(Focus, self).__init__()
        # print("c1 * 4, c2, k", c1 * 4, c2, k)
        # try:
        # pad = nf%3
        # pad_temp = 0 if pad==0 else 3-pad
        self.conv3d = nn.Conv3d(in_channels=c1*4, 
                                out_channels=c1*4,
                                kernel_size=(1,1,1),
                                stride=(1,1,1),
                                padding=(0,0,0),
                                bias=False                                  
                                )
        self.conv = Conv(c1*4, c2, k, s, padding=model_modules_shared.autopad(k,None), bias=False, num_frames=nf) #tadaconv
        # except:
        #     self.conv = Conv(c1 * 4 * nf, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # print("Focus inputs shape", x.shape)
        # print()
        # out=x
        out = self.conv3d(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # out = self.linear(out)
        # out = self.norm(out)
        # Need to check the shape here again
        return self.conv(out)
        # return self.conv(self.contract(x))        

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, num_frames=None):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, num_frames=num_frames)
        # try:
        #     self.cv2 = Conv(c_, c2, 3, 1, g=g)
        # except:
        self.cv2 = Conv(c_, c2, [1, 3, 3], 1, groups=g, num_frames=num_frames)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Add2(nn.Module):
    #  x + transformer[0] or x + transformer[1]
    def __init__(self, c1, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        if self.index == 0:
            return torch.add(x[0], x[1][0].reshape(x[0].shape))
        elif self.index == 1:
            return torch.add(x[0], x[1][1].reshape(x[0].shape))

class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13), num_frames=None):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        if num_frames is None:
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        else:
            self.cv1 = Conv(c1, c_, 1, 1, num_frames=num_frames)
            self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1, num_frames=num_frames)
            
        self.m = nn.ModuleList([nn.MaxPool3d(kernel_size=(1,x,x), stride=(1,1,1), padding=(0,x // 2,x // 2)) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class SPPv2(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    # Elimnate randomness in gradient by maxpool2d, Yes but other gradients not
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPPv2, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        b, c, f, h, w = x.shape
        x_list_fea = [m( x.reshape(b, c*f, h, w) ) for m in self.m]
        return self.cv2(torch.cat([x] + [x.reshape(b, c, f, h, w) for x in x_list_fea], 1))


class SPPv3(nn.Module):
    # Does this elimnate randomness in gradients by maxpool3d? No
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPPv3, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.k = k

    def forward(self, x):
        x = self.cv1(x)
        x_list_feat = [F.max_pool3d(x, kernel_size=(1,k,k), stride=(1,1,1), padding=(0,k // 2, k // 2), return_indices=False) for k in self.k]
        return self.cv2(torch.cat([x] + x_list_feat, 1))
        