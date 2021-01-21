import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from tools.distributed import reduce_tensor
from tools.utils import AverageMeter, to_scalar, time_str


def batch_trainer(cfg, args, epoch, model, model_ema, train_loader, criterion, optimizer, loss_w=[1, ]):
    model.train()
    epoch_time = time.time()
    loss_meter = AverageMeter()

    batch_num = len(train_loader)
    gt_list = []
    preds_probs = []
    imgname_list = []

    lr = optimizer.param_groups[1]['lr']

    for step, (imgs, gt_label, imgname) in enumerate(train_loader):

        batch_time = time.time()
        imgs, gt_label = imgs.cuda(), gt_label.cuda()
        train_logits, feat = model(imgs, gt_label)
        loss_list = criterion(train_logits, gt_label, feat, epoch)
        train_loss = loss_list[0]
        optimizer.zero_grad()
        train_loss.backward()
        if cfg.TRAIN.CLIP_GRAD:
            clip_grad_norm_(model.parameters(), max_norm=10.0)  # make larger learning rate works
        optimizer.step()
        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()

        if args.distributed:
            reduced_loss = reduce_tensor(train_loss.data, args.world_size)
            loss_meter.update(to_scalar(reduced_loss))
        else:
            loss_meter.update(to_scalar(train_loss))

        gt_list.append(gt_label.cpu().numpy())

        train_probs = train_logits.sigmoid()

        preds_probs.append(train_probs.detach().cpu().numpy())

        imgname_list.append(imgname)

        log_interval = 20

        if (step + 1) % log_interval == 0 or (step + 1) % len(train_loader) == 0:
            if args.local_rank == 0:
                print(f'{time_str()}, Step {step}/{batch_num} in Ep {epoch}, {time.time() - batch_time:.2f}s ',
                      f'train_loss:{loss_meter.avg:.4f}')

    train_loss = loss_meter.avg

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)

    if args.local_rank == 0:
        print(f'Epoch {epoch}, LR {lr}, Train_Time {time.time() - epoch_time:.2f}s, Loss: {loss_meter.avg:.4f}')

    return train_loss, gt_label, preds_probs, imgname_list


def valid_trainer(args, model, valid_loader, criterion, loss_w=[1, ]):
    model.eval()
    loss_meter = AverageMeter()
    cls_l_meter = AverageMeter()
    inter_l_meter = AverageMeter()
    intra_l_meter = AverageMeter()

    preds_probs = []
    gt_list = []
    imgname_list = []

    with torch.no_grad():
        for step, (imgs, gt_label, imgname) in enumerate(tqdm(valid_loader)):
            imgs = imgs.cuda()
            gt_label = gt_label.cuda()
            gt_list.append(gt_label.cpu().numpy())
            gt_label[gt_label == -1] = 0
            valid_logits, feat = model(imgs, gt_label)
            loss_list = criterion(valid_logits, gt_label, feat)

            valid_loss = loss_list[0]
            valid_probs = torch.sigmoid(valid_logits)
            preds_probs.append(valid_probs.cpu().numpy())

            if args.distributed:
                if len(loss_list) > 1:
                    cls_l_meter.update(to_scalar(reduce_tensor(loss_list[0], args.world_size)))
                    inter_l_meter.update(to_scalar(reduce_tensor(loss_list[1], args.world_size)))
                loss_meter.update(to_scalar(reduce_tensor(valid_loss, args.world_size)))
            else:
                if len(loss_list) > 1:
                    cls_l_meter.update(to_scalar(loss_list[0]))
                    inter_l_meter.update(to_scalar(loss_list[1]))
                loss_meter.update(to_scalar(valid_loss))

            torch.cuda.synchronize()

            imgname_list.append(imgname)

    valid_loss = loss_meter.avg

    if args.local_rank == 0:
        print(f'cls_loss:{cls_l_meter.avg:.4f}, inter_loss:{inter_l_meter.avg:.4f}, '
              f'intra_loss:{intra_l_meter.avg:.4f}')

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)

    return valid_loss, gt_label, preds_probs, imgname_list
