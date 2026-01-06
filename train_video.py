import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test_video as test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.yolo_test import Model
from utils.autoanchor import check_anchors
from utils.datasets_vid import create_dataloader_rgb_ir
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel, intersect_dicts_tadaconv, intersect_dicts_full
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume

logger = logging.getLogger(__name__)

from utils.datasets import RandomSampler
import global_var
from datetime import datetime
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ptflops import get_model_complexity_info

def train_rgb_ir(hyp, opt, device, tb_writer=None):
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'
    layers_with_grad_nan = save_dir / 'layers_with_grad_nan.txt'
    which_epoch_is_best = save_dir / 'best_model_epoch.txt'
    current_model = str(wdir / 'cur_{}.pt')
    metrics_to_check = str(save_dir / 'saved_metrics.json')
    saved_metrics = defaultdict(list)

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)

    # Configure
    plots = not opt.evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.safe_load(f)  # data dict
    is_coco = opt.data.endswith('coco.yaml')

    # Logging- Doing this before checking the dataset. Might update data_dict
    loggers = {'wandb': None}  # loggers dict
    if rank in [-1, 0]:
        opt.hyp = hyp  # add hyperparameters
        run_id = torch.load(weights).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
        wandb_logger = WandbLogger(opt, save_dir.stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict
        if wandb_logger.wandb:
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # WandbLogger might update weights, epochs if resuming

    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Model
    pretrained = weights.endswith('.pt') or opt.thermal_weights.endswith('.pt') or opt.rgb_weights.endswith('.pt')
    frames = opt.lframe+opt.gframe
    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # download if not found locally
    
        # NJOTWANI
        if opt.use_tadaconv:
            model = Model(opt.cfg, ch=3*frames, numframes=frames, nc=nc, anchors=hyp.get('anchors'), use_tadaconv=opt.use_tadaconv).to(device)  # create
        else:
            model = Model(opt.cfg, ch=3, numframes=frames, nc=nc, anchors=hyp.get('anchors'), use_tadaconv=opt.use_tadaconv).to(device)  # create    
        
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
        # import pdb; pdb.set_trace()
        if opt.load_whole_model:
            #full state is in rgb_state_dict but to keep conistent with format 
            ckpt = torch.load(opt.weights, map_location=device)  # load checkpoint
            state_dict = ckpt['model'].float().state_dict()  # to FP32
            rgb_state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)
            ir_state_dict = {}
            head_state_dict = {}
            
        elif opt.use_mode_spec_back_weights:
            ckpt = torch.load(opt.rgb_weights, map_location=device)  # load checkpoint
            ckpt_thermal = torch.load(opt.thermal_weights, map_location=device)  # load checkpoint
            state_dict_rgb = ckpt['model'].float().state_dict()  # to FP32
            state_dict_thermal = ckpt_thermal['model'].float().state_dict()  # to FP32
            
            rgb_state_dict = intersect_dicts_full(state_dict_rgb, model.state_dict(), mode ='rgb', tadaconv=opt.use_tadaconv)
            ir_state_dict = intersect_dicts_full(state_dict_thermal, model.state_dict(), mode ='ir', tadaconv=opt.use_tadaconv)
            
            if opt.detector_weights == 'thermal':
                head_state_dict = intersect_dicts_full(state_dict_thermal, model.state_dict(), back_or_head='head', tadaconv=opt.use_tadaconv)    
            elif opt.detector_weights == 'rgb':
                head_state_dict = intersect_dicts_full(state_dict_rgb, model.state_dict(), back_or_head='head', tadaconv=opt.use_tadaconv)
            elif opt.detector_weights == 'both':
                head_state_dict_RGB = intersect_dicts_full(state_dict_rgb, model.state_dict(), back_or_head='headRGB', tadaconv=opt.use_tadaconv)
                head_state_dict_Thermal = intersect_dicts_full(state_dict_thermal, model.state_dict(), back_or_head='headThermal', tadaconv=opt.use_tadaconv)
                head_state_dict = {**head_state_dict_RGB, **head_state_dict_Thermal}
            elif opt.detector_weights == 'blank':
                head_state_dict = {}
            else:
                raise 
                
        else:
            ckpt = torch.load(weights, map_location=device)  # load checkpoint
            state_dict = ckpt['model'].float().state_dict()  # to FP32
            rgb_state_dict = intersect_dicts_full(state_dict, model.state_dict(), mode ='rgb', tadaconv=opt.use_tadaconv)
            ir_state_dict = intersect_dicts_full(state_dict, model.state_dict(), mode ='ir', tadaconv=opt.use_tadaconv)
            head_state_dict = intersect_dicts_full(state_dict, model.state_dict(), back_or_head='head', tadaconv=opt.use_tadaconv)
        


        state_dict = {**rgb_state_dict, **ir_state_dict, **head_state_dict}

        # import pdb; pdb.set_trace()
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)  # load
        # of rf
        quick_count = 0
        bn_quick_count = 0
        other_keys = []
        for k in missing_keys:
            quick_count = quick_count + 1 if 'rf' in k else quick_count
            bn_quick_count = bn_quick_count + 1 if '.bn_b' in k else bn_quick_count
            if '.bn_b' not in k and 'rf' not in k:
                other_keys.append(k)
    else:
        # NJOTWANI
        if opt.opt.use_tadaconv:
            model = Model(opt.cfg, ch=3*frames, numframes=frames, nc=nc, anchors=hyp.get('anchors'), use_tadaconv=opt.use_tadaconv).to(device)  # create
        else:
            model = Model(opt.cfg, ch=3, numframes=frames, nc=nc, anchors=hyp.get('anchors'), use_tadaconv=opt.use_tadaconv).to(device)  # create    
        
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    train_path_rgb = data_dict['train_rgb']
    if not opt.whole:
        test_path_rgb = data_dict['val_rgb']
    train_path_ir = data_dict['train_ir']
    if not opt.whole:
        test_path_ir = data_dict['val_ir']

    # Freeze
    freeze = []  # parameter names to freeze (full or partial)
    layers_that_can_be_frozen = [ f'.{num}.' for num in range(0,61)] 
    if opt.freeze_model:
        freeze = state_dict
    
    if opt.freeze_backbone:
        layers_that_can_be_frozen = [ f'.{num}.' for num in range(0,32)] # freeze paramater in model.0 to model.32
        
    if opt.freeze_bb_rgb_bb_ir_det_rgb:
        layers_that_can_be_frozen = [ f'.{num}.' for num in range(0,46)] 
    
    if opt.freeze_bb_rgb_bb_ir_det_ir:
        layers_that_can_be_frozen = [ f'.{num}.' for num in range(0,32)] 
        to_add = [ f'.{num}.' for num in range(46,60)]
        layers_that_can_be_frozen.extend(to_add)
        
    # import pdb; pdb.set_trace()
    # note that fusion isa t 10,17, 26
    # freeze.remove('model.10.')
    # freeze.remove('model.17.')
    # freeze.remove('model.26.')
    have_froze =0
    # freeze = set(state_dict.keys())

    # freeze_save = 'freeze_heads.txt'
    # import pdb; pdb.set_trace()
    count_named_parameters = 0 
    for k, v in model.named_parameters():
        count_named_parameters += 1
        v.requires_grad = True  # train all layers
        # if any(x in k for x in freeze):
        num_k = k.split('.')[1]
        if k in freeze and f'.{num_k}.' in layers_that_can_be_frozen:
            print('freezing %s' % k)
            v.requires_grad = False
            # with open(freeze_save, 'a') as f:
            #     f.write(f'{k}' + '\n' )
            have_froze += 1
    
    print_weight = f'{opt.rgb_weights} and {opt.thermal_weights}' if opt.detector_weights == 'both' else weights
    logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), print_weight))  # report
    logger.info('Froze %g/%g items from %s' % ( have_froze, len(model.state_dict()), print_weight))  # report
    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d) and not opt.use_tadaconv:
            pg0.append(v.weight)  # no decay
        elif isinstance(v, nn.BatchNorm3d) and opt.use_tadaconv:
            pg0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    if opt.optimizer == 'adam':
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    elif opt.optimizer == 'sgd':
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    elif opt.optimizer == 'adamw':
        optimizer = optim.AdamW(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))
    else:
        raise Exception(f"Optimizer {opt.optimizer} is not supported.")
        

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones= [6])
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict, rgb_state_dict, ir_state_dict, head_state_dict
        if opt.use_mode_spec_back_weights:
            del ckpt_thermal, state_dict_rgb, state_dict_thermal

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    # nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    if 'detectorIDs' in opt.model_config.keys() and len(opt.model_config['detectorIDs']) > 1:
        nl = len(opt.model_config['anchors']) * len(opt.model_config['detectorIDs'])  # number of detection layers (used for scaling hyp['obj'])
    else:
        nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    # print("nl", nl)
    # print("nl", nl)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # Trainloader
    dataloader, dataset = create_dataloader_rgb_ir(train_path_rgb, train_path_ir, imgsz, batch_size, gs, opt,
                                                    opt.temporal_stride, opt.lframe, opt.gframe, opt.regex_search,
                                                    hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                                    world_size=opt.world_size, workers=opt.workers,
                                                    image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '),
                                                    dataset_used=opt.dataset_used, temporal_mosaic=opt.temporal_mosaic, 
                                                    use_tadaconv=opt.use_tadaconv, supervision_signal=opt.detection_head, 
                                                    sanitized=opt.sanitized, mosaic=opt.mosaic)
    
    if not opt.whole:
        dataloadertrain_eval, _ = create_dataloader_rgb_ir(train_path_rgb, train_path_ir, imgsz, batch_size * 2, gs, opt,
                                                    opt.temporal_stride, opt.lframe, opt.gframe, opt.regex_search,
                                                    hyp=hyp, cache=opt.cache_images and not opt.notest,rank=-1, #rect=True,# rank=-1,
                                                    world_size=opt.world_size, workers=opt.workers,
                                                    pad=0.5, prefix=colorstr('val: '),
                                                    dataset_used = opt.dataset_used, is_validation=True, 
                                                    use_tadaconv=opt.use_tadaconv, supervision_signal=opt.detection_head,
                                                    sanitized=opt.sanitized) 
    if isinstance(dataset.labels, dict):
        mlc = np.concatenate(list(dataset.labels.values()), 0)[:,0].max()  # max label class
    else:
        mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
        
    nb = len(dataloader)  # number of batches
    print(mlc)
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Process 0
    if rank in [-1, 0]:
        if not opt.whole:
            testloader, testdata = create_dataloader_rgb_ir(test_path_rgb, test_path_ir, imgsz_test, batch_size * 2, gs, opt,
                                                            opt.temporal_stride, opt.lframe, opt.gframe, opt.regex_search,
                                                            hyp=hyp, cache=opt.cache_images and not opt.notest, rank=-1,# rect=True, rank=-1,
                                                            world_size=opt.world_size, workers=opt.workers,
                                                            pad=0.5, prefix=colorstr('val: '),
                                                            dataset_used = opt.dataset_used, is_validation=True, 
                                                            use_tadaconv=opt.use_tadaconv, supervision_signal=opt.detection_head,
                                                            sanitized=opt.sanitized)           
        if not opt.resume:
            if isinstance(dataset.labels, dict):
                labels = np.concatenate(list(dataset.labels.values()), 0)
            else:
                labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                plot_labels(labels, names, save_dir, loggers)
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                    # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
                    find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    if isinstance(dataset.labels, dict):
        model.class_weights = labels_to_class_weights( list( dataset.labels.values() ) , nc).to(device) * nc  # attach class weights
    else:
        model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names
    
    
    # flops, params = get_model_complexity_info(model, (6, 3, 640, 640), as_strings=True, print_per_layer_stat=True)
    # print(flops)


    # Start training
    t0 = time.time()
    nw = round(hyp['warmup_epochs'] * nb)
    # nw = min(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    # if opt.detector_weights == 'both':
    rgb_therm_idx = opt.model_config['detectorIDs'][0]
    rgb_therm_det = model.module.model[rgb_therm_idx] if is_parallel(model) else model.model[rgb_therm_idx]  # Detect() module
    compute_loss = ComputeLoss(model, dataset_used=opt.dataset_used, kl_cross=opt.kl_cross, det=rgb_therm_det)  # init loss class

    # else:
    #     compute_loss = ComputeLoss(model, dataset_used=opt.dataset_used, kl_cross=opt.kl_cross)  # init loss class
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')
    
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        if opt.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                if isinstance(dataset.labels, dict):
                    iw = labels_to_image_weights( list(dataset.labels.values()) , nc=nc, class_weights=cw)  # image weights
                else:
                    iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        if opt.use_both_labels_for_optimization:
            mloss = torch.zeros(8, device=device)  # mean losses
            logger.info(('\n' + '%10s' * 12) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'box_ir', 'obj_ir', 'cls_ir', 'kl_div', 'total', 'labels', 'img_size'))
        else:
            mloss = torch.zeros(4, device=device)  # mean losses
            logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        step = 0
        for i, (imgs, targets, targets_ir, paths, _) in pbar:  # batch -------------------------------------------------------------
            
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
            # imgs_rgb = imgs[:, :3, :, :]
            # imgs_ir = imgs[:, 3:, :, :]
            rgb_ir_split = imgs.shape[1]//2
            imgs_rgb = imgs[:, :rgb_ir_split, :, :]
            imgs_ir = imgs[:, rgb_ir_split:, :, :]

            # Reshape RGB tensor NJOTWANI
            if not opt.use_tadaconv:  
                b, c, f, h, w = imgs_rgb.shape
                imgs_rgb = imgs_rgb.permute(0, 2, 1, 3, 4).reshape(b*f, c, h, w) 
                b, c, f, h, w = imgs_ir.shape
                imgs_ir = imgs_ir.permute(0, 2, 1, 3, 4).reshape(b*f, c, h, w)


            # FQY my code 训练数据可视化
            flage_visual = global_var.get_value('flag_visual_training_dataset')
            if flage_visual:
                from torchvision import transforms
                unloader = transforms.ToPILImage()
                for num in range(batch_size):
                    image = imgs[num, :rgb_ir_split, :, :].cpu().clone()  # clone the tensor
                    image = image.squeeze(0)  # remove the fake batch dimension
                    image = unloader(image)
                    image.save('example_%s_%s_%s_color.jpg'%(str(epoch), str(i), str(num)))
                    image = imgs[num, rgb_ir_split:, :, :].cpu().clone()  # clone the tensor
                    image = image.squeeze(0)  # remove the fake batch dimension
                    image = unloader(image)
                    image.save('example_%s_%s_%s_ir.jpg'%(str(epoch), str(i), str(num)))


            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                if opt.use_tadaconv:
                    accumulate = 1
                else:
                    accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            # with torch.autograd.detect_anomaly():
            with amp.autocast(enabled=cuda):
                # import pdb; pdb.set_trace()
                pred = model(torch.cat((imgs_rgb, imgs_ir), 1), use_tadaconv=opt.use_tadaconv, numframes=frames)  # forward
                if opt.detection_head == 'lastframe':
                    if opt.use_both_labels_for_optimization:
                        loss, loss_items = compute_loss(pred, targets.to(device), targets_ir.to(device), n_roi=opt.n_roi, img_size=opt.img_size[0])  # loss scaled by batch_size
                    else:
                        loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                else:
                    raise Exception(f"Detection head: {opt.detection_head} is not supported.")                

                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            if math.isnan(loss):
                print('GOT NAN LOSS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                import pdb; pdb.set_trace()
                raise RuntimeError("ERROR: Got NaN losses {}".format(datetime.now()))
                
            # Backward
            # with torch.autograd.detect_anomaly():
            scaler.scale(loss).backward()
            
            # Optimize
            if ni % accumulate == 0:
                
                if opt.gradient_clip:
                    scaler.unscale_(optimizer)  # unscale gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.gradient_clip)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                #vanishing gradient monitoring 
                # for name, param in model.named_parameters():
                    
                #     if param.grad.isnan().any() or param.grad.isinf().any():
                #         with open(layers_with_grad_nan, 'a') as f:
                #             f.write('{}'.format(name) + '\t' + '{}/{}'.format(torch.isnan(param.grad).sum(), int(param.grad.reshape(-1).size()[0]))  + '\n' )
                # with open(layers_with_grad_nan, 'a') as f:
                #     f.write('After iteration {} at epoch {}'.format(step, epoch)  + '\n' )
                scaler.update()
                step += 1
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                if opt.use_both_labels_for_optimization:
                    s = ('%10s' * 2 + '%10.4g' * 10) % (
                        '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                else:
                    s = ('%10s' * 2 + '%10.4g' * 6) % (
                        '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # # Plot
                # if plots and ni < 3:
                #     f = save_dir / f'train_batch{ni}.jpg'  # filename
                #     Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                #     # if tb_writer:
                #     #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                #     #     tb_writer.add_graph(torch.jit.trace(model, imgs, strict=False), [])  # add model graph
                # elif plots and ni == 10 and wandb_logger.wandb:
                #     wandb_logger.log({"Mosaics": [wandb_logger.wandb.Image(str(x), caption=x.name) for x in
                #                                   save_dir.glob('train*.jpg') if x.exists()]})

            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            if not opt.whole:
                ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
                final_epoch = epoch + 1 == epochs
                if not opt.notest or final_epoch:  # Calculate mAP
                    wandb_logger.current_epoch = epoch + 1
                    
                    train_res = deepcopy(dataset.res)
                    path_map_train = test.map_image_file_to_index(train_res, opt.detection_head)
                    opt.task = 'train'
                    results_train, _, times = test.test(data_dict,
                                                    batch_size=batch_size * 2,
                                                    imgsz=imgsz,
                                                    model=ema.ema,
                                                    single_cls=opt.single_cls,
                                                    dataloader=dataloadertrain_eval,
                                                    save_dir=save_dir,
                                                    verbose=nc < 50 and final_epoch,
                                                    plots=plots and final_epoch,
                                                    wandb_logger=wandb_logger,
                                                    compute_loss=compute_loss,
                                                    is_coco=is_coco, 
                                                    opt = opt,
                                                    path_map=path_map_train)
                    
                    val_res = deepcopy(testdata.res)
                    path_map_val = test.map_image_file_to_index(val_res, opt.detection_head)
                    opt.task = 'val'
                    results, maps, times = test.test(data_dict,
                                                    batch_size=batch_size * 2,
                                                    imgsz=imgsz_test,
                                                    model=ema.ema,
                                                    single_cls=opt.single_cls,
                                                    dataloader=testloader,
                                                    save_dir=save_dir,
                                                    verbose=nc < 50 and final_epoch,
                                                    plots=plots and final_epoch,
                                                    wandb_logger=wandb_logger,
                                                    compute_loss=compute_loss,
                                                    is_coco=is_coco,
                                                    opt = opt,
                                                    path_map=path_map_val)

                # Write
                with open(results_file, 'a') as f:
                    if opt.dataset_used == 'kaist' or 'cvc14' in opt.dataset_used:
                        if opt.use_both_labels_for_optimization:
                            f.write(s + '%10.4g' * 25 % results + '\n')  # append metrics, val_loss
                        else:
                            f.write(s + '%10.4g' * 21 % results + '\n')  # append metrics, val_loss
                    else:
                        if opt.use_both_labels_for_optimization:
                            f.write(s + '%10.4g' * 11 % results + '\n')  # append metrics, val_loss
                        else:
                            f.write(s + '%10.4g' * 8 % results + '\n')  # append metrics, val_loss
                if len(opt.name) and opt.bucket:
                    os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

                # Log
                if opt.dataset_used == 'kaist' or 'cvc14' in opt.dataset_used:
                    if opt.use_both_labels_for_optimization:
                        tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss,
                                'train/box_loss_ir', 'train/obj_loss_ir', 'train/cls_loss_ir', 'train/kldiv_loss',
                                'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.75', 'metrics/mAP_0.5:0.95', #val_metrics
                                'metrics/mAP_0.5_person', 'metrics/mAP_0.5_people', 'metrics/mAP_0.5_cyclist', 'metrics/mAP_0.5_person?',
                                'metrics/mAP_0.75_person', 'metrics/mAP_0.75_people' , 'metrics/mAP_0.75_cyclist', 'metrics/mAP_0.75_person?',
                                'metrics/mAP_0.5:0.95_person', 'metrics/mAP_0.5:0.95_people', 'metrics/mAP_0.5:0.95_cyclist', 'metrics/mAP_0.5:0.95_person?',
                                'metrics/missrate',
                                'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                                'val/box_loss_ir', 'val/obj_loss_ir', 'val/cls_loss_ir', 'val/kldiv_loss',
                                'x/lr0', 'x/lr1', 'x/lr2', 
                                'metrics_train/precision', 'metrics_train/recall', 'metrics_train/mAP_0.5', 'metrics_train/mAP_0.75', 'metrics_train/mAP_0.5:0.95',
                                'metrics_train/mAP_0.5_person', 'metrics_train/mAP_0.5_people', 'metrics_train/mAP_0.5_cyclist', 'metrics_train/mAP_0.5_person?',
                                'metrics_train/mAP_0.75_person', 'metrics_train/mAP_0.75_people' , 'metrics_train/mAP_0.75_cyclist', 'metrics_train/mAP_0.75_person?',
                                'metrics_train/mAP_0.5:0.95_person', 'metrics_train/mAP_0.5:0.95_people', 'metrics_train/mAP_0.5:0.95_cyclist', 'metrics_train/mAP_0.5:0.95_person?',
                                'metrics_train/missrate'              
                                ]  # params

                        # # raise 'double check this'
                        # for tag, result in zip(tags[6:6+21], results):
                        #     saved_metrics[tag[8:]].append(result) #these result are the validation results                    
                    else:
                        tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                                'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.75', 'metrics/mAP_0.5:0.95', #val_metrics
                                'metrics/mAP_0.5_person', 'metrics/mAP_0.5_people', 'metrics/mAP_0.5_cyclist', 'metrics/mAP_0.5_person?',
                                'metrics/mAP_0.75_person', 'metrics/mAP_0.75_people' , 'metrics/mAP_0.75_cyclist', 'metrics/mAP_0.75_person?',
                                'metrics/mAP_0.5:0.95_person', 'metrics/mAP_0.5:0.95_people', 'metrics/mAP_0.5:0.95_cyclist', 'metrics/mAP_0.5:0.95_person?',
                                'metrics/missrate',
                                'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                                'x/lr0', 'x/lr1', 'x/lr2', 
                                'metrics_train/precision', 'metrics_train/recall', 'metrics_train/mAP_0.5', 'metrics_train/mAP_0.75', 'metrics_train/mAP_0.5:0.95',
                                'metrics_train/mAP_0.5_person', 'metrics_train/mAP_0.5_people', 'metrics_train/mAP_0.5_cyclist', 'metrics_train/mAP_0.5_person?',
                                'metrics_train/mAP_0.75_person', 'metrics_train/mAP_0.75_people' , 'metrics_train/mAP_0.75_cyclist', 'metrics_train/mAP_0.75_person?',
                                'metrics_train/mAP_0.5:0.95_person', 'metrics_train/mAP_0.5:0.95_people', 'metrics_train/mAP_0.5:0.95_cyclist', 'metrics_train/mAP_0.5:0.95_person?',
                                'metrics_train/missrate'              
                                ]  # params
                        
                    start = 7 if opt.use_both_labels_for_optimization else 3 #this number means picks 'metrics/precision'
                    for tag, result in zip(tags[start:start+21], results):
                        saved_metrics[tag[8:]].append(result) #these result are the validation results                    

                    with open(metrics_to_check, 'w') as f:
                        json.dump(saved_metrics, f)
                        print('*'*25 + 'Saving Metrics JSON' + '*'*25)
                else:
                    tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                            'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.75', 'metrics/mAP_0.5:0.95',
                            'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                            'x/lr0', 'x/lr1', 'x/lr2', 
                            'metrics_train/precision', 'metrics_train/recall',
                            'metrics_train/mAP_0.5', 'metrics_train/mAP_0.75', 'metrics_train/mAP_0.5:0.95']  # params
                    
                for x, tag in zip(list(mloss[:-1]) + list(results) + lr + list(results_train), tags):
                    if tb_writer:
                        tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                    if wandb_logger.wandb:
                        wandb_logger.log({tag: x})  # W&B

                # Update best mAP
                if opt.dataset_used == 'kaist' or 'cvc14' in opt.dataset_used:
                    start= 7 if opt.use_both_labels_for_optimization else 3
                    end = 24 if opt.use_both_labels_for_optimization else 21 
                    fi = fitness(np.array(results).reshape(1, -1)[:,:end], tags[start:start+end], opt.dataset_used)  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                else:
                    fi = fitness(np.array(results).reshape(1, -1), tags, opt.dataset_used)  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                # else:
                #     fi = fitness(np.array(results).reshape(1, -1), tags[3:3+21], opt.dataset_used)  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                if fi > best_fitness:
                    best_fitness = fi
                wandb_logger.end_epoch(best_result=best_fitness == fi)

                # Save model
                if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
                    ckpt = {'epoch': epoch,
                            'best_fitness': best_fitness,
                            'training_results': results_file.read_text(),
                            'model': deepcopy(model.module.state_dict() if is_parallel(model) else model.state_dict()),
                            'ema': deepcopy(ema.ema).half(),
                            'updates': ema.updates,
                            'optimizer': optimizer.state_dict(),
                            'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None
                            }

                    # Save last, best and delete
                    if opt.save_all_model_epochs:
                        torch.save(ckpt, current_model.format(epoch))
                    torch.save(ckpt, last)
                    if best_fitness == fi:
                        torch.save(ckpt, best)
                        with open(which_epoch_is_best, 'a') as file_write:
                            file_write.write(f'Finished running on Epoch {epoch}' + '\n')
                            file_write.write(f'Best Model is at Epoch {epoch}' + '\n')
                            file_write.write(50*'%' + '\n')
                    else:
                        with open(which_epoch_is_best, 'a') as file_write:
                            file_write.write(f'Finished running on Epoch {epoch}' + '\n')
                            file_write.write(50*'%' + '\n')

                    if wandb_logger.wandb:
                        if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                            wandb_logger.log_model(
                                last.parent, opt, epoch, fi, best_model=best_fitness == fi)
                    del ckpt
                    
                    if opt.save_all_model_epochs: # reduces model size each epoch
                        loc = current_model.format(epoch)
                        loc = Path(loc)
                        if loc.exists():
                            strip_optimizer(loc)
            else:
                ckpt = {'epoch': epoch,
                        # 'best_fitness': best_fitness,
                        # 'training_results': results_file.read_text(),
                        'model': deepcopy(model.module.state_dict() if is_parallel(model) else model.state_dict()),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict()
                        }
                torch.save(ckpt, last)
                if opt.save_all_model_epochs:
                    torch.save(ckpt, current_model.format(epoch))
                    loc = current_model.format(epoch)
                    loc = Path(loc)
                    if loc.exists():
                        strip_optimizer(loc)
                del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    if rank in [-1, 0]:
        # Plots
        if plots and not opt.whole:
            plot_results(save_dir=save_dir)  # save as results.png
            if wandb_logger.wandb:
                files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
                                              if (save_dir / f).exists()]})
        # Test best.pt
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))

        # Strip optimizers
        final = best if best.exists() else last  # final model
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
        
        if opt.bucket:
            os.system(f'gsutil cp {final} gs://{opt.bucket}/weights')  # upload
        if wandb_logger.wandb and not opt.evolve:  # Log the stripped model
            wandb_logger.wandb.log_artifact(str(final), type='model',
                                            name='run_' + wandb_logger.wandb_run.id + '_model',
                                            aliases=['last', 'best', 'stripped'])
        wandb_logger.finish_run()
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return results




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5l.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='./models/transformer/yolov5l_fusion_add_FLIR_aligned.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='./data/multispectral/FLIR_aligned.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', default='sgd', type=str, choices=['adam', 'sgd', 'adamw'], help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--lframe', type=int, default=6, help='Number of Local Frames in Batch')
    parser.add_argument('--gframe', type=int, default=0, help='Number of Global Frames in Batch')
    parser.add_argument('--temporal_stride', type=int, default=3, help='Local Frames in a batch are strided by this amount')
    parser.add_argument('--regex_search', type=str, default=".images...", help="For kaist:'.set...V...' , For camel use:'.images...' .This helps the dataloader seperate ordered list in indivual videos for kaist use:r'.set...V...' ")
    parser.add_argument('--dataset_used', type=str, default="kaist", help='dataset used: kaist, camel, cvc14, cvc14_align, cvc14_align_resized ')
    parser.add_argument('--temporal_mosaic', action='store_true', help='load mosaic with temporally related sequences of images')
    parser.add_argument('--mosaic', action='store_true', help='use mosaic augmentations')
    parser.add_argument('--use_tadaconv', action='store_true', help='load tadaconv as feature extractor')
    parser.add_argument('--detection_head', type=str, default='lastframe', choices=['lastframe', 'midframe', 'fullframes'], help='selects the detection head: 1) lastframe, 2) midframe, 3) fullframes')
    # parser.add_argument('--fusion_strategy', type=str, default='TadaConvSpatialGPT', help='selects the fusion strategy for thermal and RGB: 1) GPT, 2) TadaConvSpatialGPT,')
    parser.add_argument('--save_all_model_epochs', action='store_true', help='save all model epochs')
    parser.add_argument('--json-class', action='store_true', help='use class number in json instead of default 1')
    parser.add_argument('--json_gt_loc', type=str, default='./json_gt/')
    parser.add_argument('--task', type=str, default='val', help='train, val')
    parser.add_argument('--use_mode_spec_back_weights', action='store_true', help='when true load thermal weights into thermal stream and RGB weights into RGB stream')
    parser.add_argument("--thermal_weights", type=str, default='yolov5l_kaist_best_thermal.pt', help='initial thermal weights path')
    parser.add_argument("--rgb_weights", type=str, default='yolov5l_kaist_best_rgb.pt', help='initial rgb weights path')
    parser.add_argument('--detector_weights', type=str, default='thermal', help="use 1) 'thermal', 2) 'rgb' to load pretrained detector head weights")
    parser.add_argument('--sanitized', action='store_true', help='using sanitized label only')
    parser.add_argument('--gradient_clip', type=float, default=0.0, help='clip the gradient')
    parser.add_argument('--feature_visualization', action='store_true', help='visualize features')
    parser.add_argument('--all_objects', action='store_true', help='Include unaligned objects for training')
    parser.add_argument('--load_whole_model', action='store_true', help='load whole model, used to see if kaist can help cvc14')
    parser.add_argument('--ignore_high_occ', action='store_true')
    parser.add_argument('--use_both_labels_for_optimization', action='store_true', help='during optimization we use both thermal labels and rgb labels')
    parser.add_argument('--resize_cvc14_for_eval', action='store_true', help='resize bbox detected during validation from 640x512 to 640x471')
    parser.add_argument('--n_roi', type=int, default=300, help='RoI in inference for loss function computation, if set to 0 KL divergence not used')
    parser.add_argument('--select_thermal_rgb_inference', action='store_true', help='to select either RGB or thermal features in inference')
    parser.add_argument('--kl_cross', action='store_true', help='use cross modality cosine similarity')
    parser.add_argument('--freeze_model', action='store_true', help='freeze yolov5 streams, fusion and tadaconv can still train')
    parser.add_argument('--freeze_backbone', action='store_true', help='freeze backbone')
    parser.add_argument('--freeze_bb_rgb_bb_ir_det_rgb', action='store_true', help='freeze backbone and rgb head')
    parser.add_argument('--freeze_bb_rgb_bb_ir_det_ir', action='store_true', help='freeze backbone and thermal head')



    opt = parser.parse_args()

    # FQY  Flag for visualizing the paired training imgs
    global_var._init()
    global_var.set_value('flag_visual_training_dataset', False)
    opt.even_val = True if "even_val" in opt.data else False # use a more even train-val split
    opt.whole = True if "whole" in opt.data else False # use the whole dataset for training


    # Ensure yolov_modules_to_select.yaml and parser variables are equal
    with open('./models/yolov_modules_to_select.yaml') as f:
        select_modules = yaml.safe_load(f)  # data dict
    assert opt.use_tadaconv == select_modules['use_tadaconv'], "Make sure *****use_tadaconv***** is set to either BOTH True or BOTH False in the input arguments and YAML file" # as Feature Extractor
    
    with open(opt.cfg) as f:
        model_config = yaml.safe_load(f)
    
    opt.model_config = model_config
    
    if opt.detection_head == 'fullframes':
        assert 'FullFramesDetect' ==  model_config['head'][-1][2], "Make sure correct config file is used or make sure fullframes parser arg is set to False"   
    elif opt.detection_head == 'lastframe':
        assert 'Detect' == model_config['head'][-1][2] or 'LastFrameDetect' == model_config['head'][-1][2] \
                or 'LastFrameThermalRgbDetect' in model_config['head'][-1][2] or 'LastFrameDeformDetect' in model_config['head'][-1][2], "Make sure correct config file is used for lastframe detection head" 
    elif opt.detection_head == 'midframe':
        assert 'MidFrameDetect' == model_config['head'][-1][2], "Make sure correct config file is used for midframe detection head" 
    else:
        raise Exception(f"Detection head: {opt.detection_head} is not supported.")                
      
    # import pdb; pdb.set_trace()
    if opt.dataset_used == 'kaist' and not opt.whole:
        if opt.sanitized:
            if 'small' in opt.data: # for quick debugging
                train_json = opt.json_gt_loc + f'kaistsmall_train_lframe_{opt.lframe}_stride_{opt.temporal_stride}{"_whole" if opt.whole else ""}_sanitized{"_even_val" if opt.even_val else ""}{"_all_objects" if opt.all_objects else ""}{"_midframe" if opt.detection_head == "midframe" else ""}{"_ignore_high_occ" if opt.ignore_high_occ else ""}.json' 
                val_json = opt.json_gt_loc + f'kaistsmall_val_lframe_{opt.lframe}_stride_{opt.temporal_stride}{"_whole" if opt.whole else ""}_sanitized{"_even_val" if opt.even_val else ""}{"_all_objects" if opt.all_objects else ""}{"_midframe" if opt.detection_head == "midframe" else ""}{"_ignore_high_occ" if opt.ignore_high_occ else ""}.json'
            else:
                train_json = opt.json_gt_loc + f'kaist_train_lframe_{opt.lframe}_stride_{opt.temporal_stride}{"_whole" if opt.whole else ""}_sanitized{"_even_val" if opt.even_val else ""}{"_all_objects" if opt.all_objects else ""}{"_midframe" if opt.detection_head == "midframe" else ""}{"_ignore_high_occ" if opt.ignore_high_occ else ""}.json'
                val_json = opt.json_gt_loc + f'kaist_val_lframe_{opt.lframe}_stride_{opt.temporal_stride}{"_whole" if opt.whole else ""}_sanitized{"_even_val" if opt.even_val else ""}{"_all_objects" if opt.all_objects else ""}{"_midframe" if opt.detection_head == "midframe" else ""}{"_ignore_high_occ" if opt.ignore_high_occ else ""}.json'
            print(train_json)
            assert os.path.isfile(train_json) == True, "Make sure to gt generate json file for train, see kaist_to_json.py"
            assert (opt.whole) or (os.path.isfile(val_json) == True), "Make sure to gt generate json file for validation, see kaist_to_json.py"
        else:
            if 'small' in opt.data: # for quick debugging
                train_json = opt.json_gt_loc + f'kaistsmall_train_lframe_{opt.lframe}_stride_{opt.temporal_stride}.json' 
                val_json = opt.json_gt_loc + f'kaistsmall_val_lframe_{opt.lframe}_stride_{opt.temporal_stride}.json'
            else:
                train_json = opt.json_gt_loc + f'kaist_train_lframe_{opt.lframe}_stride_{opt.temporal_stride}.json'
                val_json = opt.json_gt_loc + f'kaist_val_lframe_{opt.lframe}_stride_{opt.temporal_stride}.json'
            assert os.path.isfile(train_json) == True, "Make sure to gt generate json file for train, see kaist_to_json.py"
            assert os.path.isfile(val_json) == True, "Make sure to gt generate json file for validation, see kaist_to_json.py"
    # Need to add insert

    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
    if opt.global_rank in [-1, 0]:
        # check_git_status()
        check_requirements()

    # Resume
    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = \
            check_file(opt.cfg), ckpt, True, opt.total_batch_size, *apriori  # reinstate
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve))

    # DDP mode
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)  # load hyps

    # Train
    logger.info(opt)
    if not opt.evolve:
        tb_writer = None  # init loggers
        if opt.global_rank in [-1, 0]:
            prefix = colorstr('tensorboard: ')
            logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
            tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
            import wandb
            wandb.init(mode="disabled")
            train_rgb_ir(hyp, opt, device, tb_writer)



    # # Evolve hyperparameters (optional)
    # else:
    #     # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
    #     meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
    #             'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
    #             'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
    #             'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
    #             'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
    #             'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
    #             'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
    #             'box': (1, 0.02, 0.2),  # box loss gain
    #             'cls': (1, 0.2, 4.0),  # cls loss gain
    #             'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
    #             'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
    #             'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
    #             'iou_t': (0, 0.1, 0.7),  # IoU training threshold
    #             'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
    #             'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
    #             'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
    #             'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
    #             'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
    #             'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
    #             'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
    #             'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
    #             'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
    #             'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
    #             'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
    #             'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
    #             'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
    #             'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
    #             'mixup': (1, 0.0, 1.0)}  # image mixup (probability)

    #     assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
    #     opt.notest, opt.nosave = True, True  # only test/save final epoch
    #     # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
    #     yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # save best result here
    #     if opt.bucket:
    #         os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

    #     for _ in range(300):  # generations to evolve
    #         if Path('evolve.txt').exists():  # if evolve.txt exists: select best hyps and mutate
    #             # Select parent(s)
    #             parent = 'single'  # parent selection method: 'single' or 'weighted'
    #             x = np.loadtxt('evolve.txt', ndmin=2)
    #             n = min(5, len(x))  # number of previous results to consider
    #             x = x[np.argsort(-fitness(x))][:n]  # top n mutations
    #             w = fitness(x) - fitness(x).min()  # weights
    #             if parent == 'single' or len(x) == 1:
    #                 # x = x[random.randint(0, n - 1)]  # random selection
    #                 x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
    #             elif parent == 'weighted':
    #                 x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

    #             # Mutate
    #             mp, s = 0.8, 0.2  # mutation probability, sigma
    #             npr = np.random
    #             npr.seed(int(time.time()))
    #             g = np.array([x[0] for x in meta.values()])  # gains 0-1
    #             ng = len(meta)
    #             v = np.ones(ng)
    #             while all(v == 1):  # mutate until a change occurs (prevent duplicates)
    #                 v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
    #             for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
    #                 hyp[k] = float(x[i + 7] * v[i])  # mutate

    #         # Constrain to limits
    #         for k, v in meta.items():
    #             hyp[k] = max(hyp[k], v[1])  # lower limit
    #             hyp[k] = min(hyp[k], v[2])  # upper limit
    #             hyp[k] = round(hyp[k], 5)  # significant digits

    #         # Train mutation
    #         results = train(hyp.copy(), opt, device)

    #         # Write mutation results
    #         print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

    #     # Plot results
    #     plot_evolution(yaml_file)
    #     print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
    #           f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')
