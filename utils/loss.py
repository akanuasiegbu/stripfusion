# Loss functions

import torch
import torch.nn as nn

from utils.general import bbox_iou
from utils.torch_utils import is_parallel
import torchvision.ops.roi_align as roi_align
from torchvision.ops import box_iou

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False, dataset_used=None, kl_cross=False, det = None):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters
        
        if not det:
        
            # det refers to the detection layer of the model, which is often the final layer in YOLO-style object detection models
            det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
        
        # det.nl refers to the number of output layers (detection layers) in the detection model.
        # These values are balancing weights applied to losses computed for different detection layers.
        # They assign different importance to losses from different layers.
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7 
        
        self.ssi = list(det.stride).index(16) if autobalance else 0 
        
        # The self.autobalance variable is used to control whether the loss balancing between 
        # different layers in the detection model is done automatically during training       
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))
            
        # batchmean means that after computing the element-wise KL Divergence for each 
        # sample in the batch, the losses are summed across all elements, and then 
        # averaged across the batch size
        self.kl_loss = nn.KLDivLoss(reduction="batchmean") 
        
        self.dataset_used = dataset_used
        self.kl_cross = kl_cross
        # nn.KLDivLoss()

    def __call__(self, p, targets, targets_ir=None, n_roi=None, img_size=None, inference=False): 
        """
        n_roi: RoI for kl divergence calc
        img_size: img size assuming imgage rescale to img_size x img_size, should be integar 
        """
        device = targets.device
        self.n_roi  = n_roi
        
        # used for normalizing bounding box coordinates
        self.img_size =  img_size
        
        self.iou_sum = []
        assert self.img_size is None or type(self.img_size) is int

        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        lkldiv = torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p[:3], targets)  # targets
        
        # It takes the first three layers of prediction from the RGB branch  and the corresponding ground truth
        #first 3 in predictions are from RGB head, 80x80,40x40,20x20
        lbox, lobj, lcls, bs, obji_list, tobj_found = self.calc_losses(p[:3], tcls, tbox, indices, anchors, device,
                                                lcls, lbox, lobj, inference)
        
        if targets_ir is not None:
            lcls_ir, lbox_ir, lobj_ir = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
            tcls_ir, tbox_ir, indices_ir, anchors_ir = self.build_targets(p[3:6], targets_ir)  # targets_ir
            
            # next 3 in predictions are from thermal head
            lbox_ir, lobj_ir, lcls_ir, bs_ir, obji_list_ir, tobj_found_ir = self.calc_losses(p[3:6], tcls_ir, tbox_ir, indices_ir, anchors_ir, device,
                                                                lcls_ir, lbox_ir, lobj_ir, inference)

        if inference:
            return self.iou_sum # use for reliability per image
        
        if self.n_roi and targets_ir is not None:
            lkldiv = self.kl_div(p,self.iou_sum[:3], self.iou_sum[3:], tobj_found, tobj_found_ir, lkldiv)

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        
        #Scaling losses by their respective hyperparameters:
        lbox *= self.hyp['box'] 
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        
        # bs = tobj.shape[0]  # batch size
        if targets_ir is not None:
            lbox_ir *= self.hyp['box']
            lobj_ir *= self.hyp['obj']
            lcls_ir *= self.hyp['cls']

        if self.n_roi and targets_ir is not None:
            lkldiv *= self.hyp['kldiv']

        if targets_ir is not None:
            loss = lbox + lobj + lcls + lbox_ir + lobj_ir+ lcls_ir + lkldiv
            
            # the total loss (lbox + lobj + lcls) is multiplied by the batch size (bs). 
            # This ensures that the final loss is properly weighted by the number of images processed in the batch.
            return (lbox + lobj + lcls)*bs + (lbox_ir + lobj_ir+ lcls_ir)*bs_ir + lkldiv , torch.cat((lbox, lobj, lcls, lbox_ir, lobj_ir, lcls_ir, lkldiv, loss)).detach()
        else:
            loss = lbox + lobj + lcls
            return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach() 

    def calc_losses(self, p, tcls, tbox, indices, anchors, device, lcls, lbox, lobj, inference):
        """
        Losses (bounding box, objectness, and classification) for one branch (thermal and RGB) are calculated 
        Inputs:
            p: 80x80, 40x40 or 20x20 tensor for detection 
            tcls: from build_targets
            tbox: from build_targets
            indices: from build_targets
            anchors: from build_targets
            device: cpu or gpu
            lbox: loss values for bounding boxes 
            lobj: objectness loss
            lcls: classification loss
            
        
        Ouputs:
            lbox: Updated loss values for bounding boxes 
            lobj: updated objectness loss
            lcls: updated classification loss
            obji_list: Objectness scores for the a specfic modalities (rgb or thermal) branch.
            tobj_found: Indices of the objects detected.
        """
        obji_list = []
        tobj_found = []
        for i, pi in enumerate(p):  # layer index (i), layer predictions (pi)
            jj = i % 3 #so that RGB and thermal losses are treated equally in theory by the self.balance 
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                
                # when max found self.iou_sum updates inside get_ciou_mean_top_k (subfunction call of find_max)
                tobj_found.append(self.find_max(tobj)) if self.n_roi else tobj_found.append([])
                
                # averages top N bounding boxes found to get reliability of that modality 
                look_here =  tobj_found[-1]
                self.get_ciou_mean_top_k(tobj, look_here, inference) if self.n_roi else 0

                
                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
            else:
                tobj_found.append([])
                self.iou_sum.append(0)


            obji = self.BCEobj(pi[..., 4], tobj)
            obji_list.append(obji.detach())
            lobj += obji * self.balance[jj]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
                raise 'need to fix/doublecheck indexing if self.autobalance is True'
        bs = tobj.shape[0]  # batch size
        return lbox, lobj, lcls, bs, obji_list, tobj_found

    def kl_div(self, p,obji_list, obji_list_ir, tobj_found, tobj_found_ir, lkldiv, scale_offset=3):
        # import pdb; pdb.set_trace()
        len_p = len(p)
        
        # last "prediction" is features from both rgb and thermal branch
        batch, _, _, _,_ = p[6][0][0].shape

        is_rgb_per_scale = []
        for rgb_score, ir_score in zip(obji_list, obji_list_ir):
            
            #creates a bool where true when rgb_score greater for each scale
            is_rgb_per_scale.append(rgb_score>ir_score) 

        xywh_list = []
        batch_belongs = torch.zeros((batch*self.n_roi,1))
        for i in range(batch):
            batch_belongs[self.n_roi*i:self.n_roi*(i+1), :] = i
        
        # Need to create the RoI in normalized coordinates
        for list_i, is_rgb in enumerate(is_rgb_per_scale):
            
            # Each image pair has n_roi that can be used
            xywh = torch.zeros(batch*self.n_roi,4)
            if is_rgb:
                
                # use the coordinates from the RGB branch
                for ele_j in range(batch*self.n_roi):
                    if tobj_found[list_i] == []:
                        break
                    fi, se, th, fo = tobj_found[list_i][ele_j]
                    xywh[ele_j,:] = p[list_i][fi,se,th,fo,:4]
            else:
                # use the cooridnates from the Thermal branch
                for ele_j in range(batch*self.n_roi):
                    if tobj_found_ir[list_i] == []:
                        break
                    fi, se, th, fo = tobj_found_ir[list_i][ele_j]
                    
                    # using 'list_i + 3' is so that we use coordinates from thermal branch
                    xywh[ele_j,:] = p[list_i+3][fi,se,th,fo,:4]
                    
            xywh_detached = xywh.detach()
            xywh_list.append(torch.cat((batch_belongs,xywh_detached), dim=1) if torch.nonzero(xywh_detached).shape[0] > 0 else [])
        
        prob_rgb, prob_ir, prob_rgb_ir = [], [], []
        
        #  that are in bounds, assume that image is resized to img_size*img_size currently
        im_size = torch.ones(batch*self.n_roi,4)
        im_size[:,0],im_size[:,1] = 0, 0
        im_size[:,2],im_size[:,3] = self.img_size -1, self.img_size-1
        
        
        size_info = []
        for muiti_scale, xywh in enumerate(xywh_list):
            if xywh == []:
                prob_rgb.append([])
                prob_ir.append([])
                prob_rgb_ir.append([])
                continue
            
            rgb_feat = p[6][muiti_scale][0]
            ir_feat = p[6][muiti_scale][1]
            b, c, f, h,w = rgb_feat.shape
            size_info.append([b,c,f,h,w])
            
            
            xywh[:,3] = xywh[:,1] + xywh[:,3]  # x2 = x1 + w
            xywh[:,3] = xywh[:,2] + xywh[:,4]  # y2 = y1 + h
            
            # scales last four [1:] to image size (0th index corresponds to batch index so skipped)
            xywh[:,1:] = xywh[:,1:]*self.img_size -1
            tlbr = xywh
            
            # Keep detections that overlap with the image size
            ious = box_iou(im_size, tlbr[:,1:]) 
            ious = torch.diagonal(ious, 0)
            to_keep = ious > 0

            # further making sure that boxes kept stay fullly within image boundaries
            tlbr = torch.clamp(tlbr, min=0, max=self.img_size-1) 
            
            # create random roi that are actually within the image to fill the bbox that fall outside the image
            tlbr[~to_keep, 1:3] = torch.randint(low=0, high=self.img_size-11, size=(sum(~to_keep),2)).to(tlbr.dtype)
            tlbr[~to_keep, 3:] = tlbr[~to_keep, 1:3] + 10
            
            rgb = roi_align(input=rgb_feat.reshape(b,-1,h,w), boxes = tlbr.cuda(), output_size= (3,3), spatial_scale = h / self.img_size , aligned=True)
            ir = roi_align(input=ir_feat.reshape(b,-1,h,w), boxes = tlbr.cuda(), output_size= (3,3), spatial_scale = h / self.img_size, aligned=True )
            
            # ( K, frames*c, 3, 3) ->  (b, K, c*frames*3*3)
            rgb_reshaped = rgb.reshape(b, -1, c*f*9)
            ir_reshaped = ir.reshape(b, -1, c*f*9)
            
            # want to do cosine similarity on each element in each frame
            rgb_norm = rgb_reshaped / rgb_reshaped.norm(dim=2)[:,:,None]
            ir_norm = ir_reshaped / ir_reshaped.norm(dim=2)[:,:,None]
            cos_rgb = torch.matmul(rgb_norm, rgb_norm.transpose(1,2))
            cos_ir = torch.matmul(ir_norm, ir_norm.transpose(1,2))

            if self.kl_cross:
                cos_rgb_ir = torch.matmul(rgb_norm, ir_norm.transpose(1,2))
                rgb_ir_exp = torch.exp(cos_rgb_ir)
                prob_rgb_ir.append(rgb_ir_exp / torch.sum(rgb_ir_exp, dim=2)[:,:, None])

            
            rgb_exp = torch.exp(cos_rgb)
            ir_exp = torch.exp(cos_ir)
            prob_rgb.append(rgb_exp / torch.sum(rgb_exp, dim=2)[:,:, None])
            prob_ir.append(ir_exp / torch.sum(ir_exp, dim=2)[:,:, None])
            

        for list_i, is_rgb in enumerate(is_rgb_per_scale):
            if prob_ir[list_i] == [] or prob_rgb[list_i] == []:
                continue
            
            if is_rgb:#  meaning rgb is more reliable
                # means rgb had more objects on average for that scale : KL (RGB || IR)
                lkldiv += self.balance[list_i]*self.kl_loss(prob_ir[list_i].log(), prob_rgb[list_i]) #loss expects the argument input in the log-space.
            else:
                # means ir had more objects on average for that scale : KL ( IR || RGB)
                lkldiv += self.balance[list_i]*self.kl_loss(prob_rgb[list_i].log(), prob_ir[list_i]) #loss expects the argument input in the log-space.
        return lkldiv
   
    def find_max(self, tobj):
        b,a,gx, gy = tobj.shape
        find_max = tobj.reshape(b, a*gx*gy)
        ele, indices = torch.sort(find_max)
        first_index = torch.zeros(b,self.n_roi,1).cuda()
        for i in range(b):
            first_index[i,:,:] = i

        second_index = indices[:,-self.n_roi:] // (gx*gy)
        
        third_index =  ( indices[:,-self.n_roi:] % (gx*gy) ) // gx

        fourth_index = ( indices[:,-self.n_roi:] % (gx*gy) ) % gy

        look_here = torch.cat((first_index, second_index.unsqueeze(-1),
                               third_index.unsqueeze(-1),
                                fourth_index.unsqueeze(-1)), -1).to(torch.int64)
        
        look_here = look_here.reshape(b,self.n_roi,4).reshape(-1,4)
        # self.get_ciou_mean_top_k(tobj, look_here)

        return look_here
    
    def get_ciou_mean_top_k(self, tobj, look_here, inference):
        if not inference:
            get_mean = []
            for i in range(len(look_here)):
                fi, se, th, fo = look_here[i]
                get_mean.append(tobj[fi,se,th,fo] )
            self.iou_sum.append(sum(get_mean)/len(get_mean))
        else:
            b, _,_,_ = tobj.shape
            batch_index = look_here.reshape(b, self.n_roi, 4)
            for b_index in range(batch_index.shape[0]):
                one_batch_cious = []
                for n_roi in range(batch_index.shape[1]):
                    fi, se, th, fo = batch_index[b_index, n_roi]
                    one_batch_cious.append(tobj[fi,se,th,fo].item() )
                self.iou_sum.append(sum(one_batch_cious)/batch_index.shape[1])
                    
                    
            
        # return get_mean
        
    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            jj = i%3 # use the same anchors for rgb and thermal
            anchors = self.anchors[jj]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
