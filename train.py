"""
Training script for CTD-YOLO
"""
import argparse
import os
import sys
import time
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.ctd_yolo import CTD_YOLO
from models.losses import ComputeLoss
from models.maa import ModelMAA
from utils import colorstr, increment_path, check_img_size


def train(opt):
    """Train CTD-YOLO model"""
    save_dir = Path(opt.save_dir)
    epochs, batch_size, weights = opt.epochs, opt.batch_size, opt.weights
    device = torch.device(opt.device)
    
    # Directories
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters
    with open(opt.hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)

    # Loggers
    print(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Model
    check_suffix = weights.endswith('.pt')
    pretrained = check_suffix and weights != ''
    if pretrained:
        ckpt = torch.load(weights, map_location='cpu')
        model = CTD_YOLO(opt.cfg or ckpt['model'].yaml, ch=3, nc=opt.nc, anchors=hyp.get('anchors')).to(device)
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []
        csd = ckpt['model'].float().state_dict()
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)
        model.load_state_dict(csd, strict=False)
        print(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')
    else:
        model = CTD_YOLO(opt.cfg, ch=3, nc=opt.nc, anchors=hyp.get('anchors'), use_cdm=opt.use_cdm).to(device)
    
    # Load CDM pretrained weights
    if opt.use_cdm and opt.cdm_weights and Path(opt.cdm_weights).exists():
        print(f'Loading CDM weights from {opt.cdm_weights}')
        cdm_ckpt = torch.load(opt.cdm_weights, map_location='cpu')
        if hasattr(model, 'cdm'):
            model.cdm.load_state_dict(cdm_ckpt['model_state_dict'], strict=False)
            print('CDM weights loaded successfully')

    # Freeze layers
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]
    for k, v in model.named_parameters():
        v.requires_grad = True
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False

    # Image size
    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)

    # Batch size
    if batch_size == -1:
        batch_size = check_train_batch_size(model, imgsz, amp=True)
        opt.batch_size = batch_size

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    # Use Adam optimizer as specified in requirements
    optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    
    # Scheduler - Reduce on Plateau strategy
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',  # maximize validation mAP
        factor=hyp.get('reduce_lr_factor', 0.5),
        patience=hyp.get('reduce_lr_patience', 3),
        min_lr=hyp.get('min_lr', 1e-6),
        verbose=True
    )

    # MAA
    maa = ModelMAA(model) if opt.maa else None

    # Resume
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']
        if maa and ckpt.get('maa'):
            maa.maa.load_state_dict(ckpt['maa'].float().state_dict())
            maa.updates = ckpt['updates']
        if opt.resume:
            start_epoch = ckpt['epoch'] + 1
        del ckpt, csd

    # Data loading
    with open(opt.data, errors='ignore') as f:
        data_dict = yaml.safe_load(f)
    
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = int(data_dict['nc'])
    names = data_dict['names']
    
    # Trainloader
    train_loader = create_dataloader(train_path, imgsz, batch_size, gs, opt, hyp=hyp, augment=True,
                                   cache=None if opt.cache == 'val' else opt.cache, rect=opt.rect,
                                   rank=-1, workers=opt.workers, image_weights=opt.image_weights,
                                   quad=opt.quad, prefix=colorstr('train: '), shuffle=True)[0]

    # Process 0
    if opt.val:
        val_loader = create_dataloader(val_path, imgsz, batch_size // 2, gs, opt, hyp=hyp, cache=None if opt.noval else opt.cache,
                                     rect=True, rank=-1, workers=opt.workers * 2, pad=0.5,
                                     prefix=colorstr('val: '))[0]

    # Model attributes
    nl = de_parallel(model).model[-1].nl  # number of detection layers
    hyp['box'] *= 3 / nl  # scale to layers
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(train_loader.dataset.labels, nc).to(device) * nc

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * len(train_loader)), 100)  # number of warmup iterations
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    stopper, stop = EarlyStopping(patience=opt.patience), False
    compute_loss = ComputeLoss(model, use_wiou=True)  # init loss class with WIOU

    print(f'Image sizes {imgsz} train, {imgsz} val\n'
          f'Using {train_loader.num_workers * 2} dataloader workers\n'
          f"Logging results to {colorstr('bold', save_dir)}\n"
          f'Starting training for {epochs} epochs...')

    for epoch in range(start_epoch, epochs):
        model.train()

        # Update image weights (optional, single-GPU only)
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc
            iw = labels_to_image_weights(train_loader.dataset.labels, nc=nc, class_weights=cw)
            train_loader.dataset.indices = random.choices(range(train_loader.dataset.n), weights=iw, k=train_loader.dataset.n)

        mloss = torch.zeros(3, device=device)  # mean losses
        pbar = enumerate(train_loader)
        print(('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss', 'Instances', 'Size'))
        pbar = tqdm(pbar, total=len(train_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        optimizer.zero_grad()
        
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + len(train_loader) * epoch  # number integrated batches
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with torch.cuda.amp.autocast(enabled=True):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                last_opt_step = ni
            if maa:
                maa.update(model)

            # Log
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                               (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))

        # Validation
        if opt.val:
            results, maps, _ = validate.run(data_dict,
                                          batch_size=batch_size // 2,
                                          imgsz=imgsz,
                                          model=maa.maa if maa else model,
                                          single_cls=opt.single_cls,
                                          dataloader=val_loader,
                                          save_dir=save_dir,
                                          plots=False,
                                          compute_loss=compute_loss)

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        stop = stopper(epoch=epoch, fitness=fi)  # early stop check
        if fi > best_fitness:
            best_fitness = fi

        # Scheduler - step with validation metric for ReduceLROnPlateau
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        if opt.val:
            scheduler.step(fi)  # step with fitness metric
        else:
            scheduler.step(0)  # step with dummy metric if no validation

        # Save model
        if (not opt.nosave) or (epoch == epochs - 1):  # if save
            ckpt = {
                'epoch': epoch,
                'best_fitness': best_fitness,
                'model': deepcopy(de_parallel(model)).half(),
                'maa': deepcopy(maa.maa).half() if maa else None,
                'updates': maa.updates if maa else None,
                'optimizer': optimizer.state_dict(),
                'date': datetime.now().isoformat()}

            # Save last, best and delete
            torch.save(ckpt, last)
            if best_fitness == fi:
                torch.save(ckpt, best)
            del ckpt

        # Stop training
        if stop:
            break

    # End training
    print(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/ctd_yolo_config.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/ce-cctsdb.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train,val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    parser.add_argument('--maa', action='store_true', help='use MAA')
    parser.add_argument('--val', action='store_true', help='validate during training')
    parser.add_argument('--nc', type=int, default=58, help='number of classes')
    parser.add_argument('--use-cdm', action='store_true', help='use CDM preprocessing')
    parser.add_argument('--cdm-weights', type=str, default='', help='CDM pretrained weights path')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt):
    # Checks
    if opt.resume and not check_wandb_resume(opt):
        opt.resume = False

    # Parameters
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # Train
    train(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)


# Helper functions (simplified versions)
def intersect_dicts(da, db, exclude=()):
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}

def one_cycle(y1=0.0, y2=1.0, steps=100):
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

def fitness(x):
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)

def de_parallel(model):
    return model.module if hasattr(model, 'module') else model

# Placeholder functions - would need full implementation
def create_dataloader(*args, **kwargs):
    pass

def check_train_batch_size(*args, **kwargs):
    return 16

def labels_to_class_weights(*args, **kwargs):
    return torch.ones(58)

def labels_to_image_weights(*args, **kwargs):
    return [1] * 1000

def check_wandb_resume(*args, **kwargs):
    return False

class EarlyStopping:
    def __init__(self, patience=30):
        self.patience = patience
        self.count = 0
        self.best_fitness = 0
        
    def __call__(self, epoch, fitness):
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.count = 0
        else:
            self.count += 1
        return self.count >= self.patience

# Import required modules
import math
import random
import numpy as np
from copy import deepcopy
from datetime import datetime
import validate
