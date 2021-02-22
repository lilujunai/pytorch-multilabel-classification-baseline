import argparse
import os
import pickle
from collections import defaultdict
from datetime import datetime

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from visdom import Visdom

from configs import cfg, update_config

from dataset.multi_label.coco import COCO14
from dataset.augmentation import get_transform
from metrics.ml_metrics import get_map_metrics, get_multilabel_metrics
from models.model_ema import ModelEmaV2
from models.tresnet.tresnet import TResnetL
from optim.adamw import AdamW
from scheduler.cosine_lr import CosineLRScheduler
from tools.distributed import distribute_bn
import torch.distributed as dist

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.data import DataLoader

from batch_engine import valid_trainer, batch_trainer
from models.base_block import FeatClassifier
from models.model_factory import model_dict, classifier_dict, loss_dict

from tools.function import get_model_log_path, get_reload_weight, seperate_weight_decay
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, str2bool

# from thop import profile, clever_format
from ptflops import get_model_complexity_info

set_seed(605)


# torch.backends.cudnn.benchmark = True


# torch.autograd.set_detect_anomaly(True)
def main(cfg, args):
    exp_dir = os.path.join('exp_result', cfg.DATASET.NAME)
    # model_name = cfg.NAME
    # if not cfg.use_bn:
    #     model_name += '_wobn'

    model_dir, log_dir = get_model_log_path(exp_dir, cfg.NAME)
    stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')
    save_model_path = os.path.join(model_dir, f'ckpt_max_{time_str()}.pth')

    visdom = None
    if cfg.VIS.VISDOM:
        visdom = Visdom(env=f'{cfg.DATASET.NAME}_' + cfg.NAME, port=8401)
        assert visdom.check_connection()

    writer = None
    if cfg.VIS.TENSORBOARD.ENABLE:
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        writer_dir = os.path.join(exp_dir, cfg.NAME, 'runs', current_time)
        writer = SummaryWriter(log_dir=writer_dir)

    if cfg.REDIRECTOR:
        if args.local_rank == 0:
            print('redirector stdout')
        ReDirectSTD(stdout_file, 'stdout', False)

    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    else:
        args.distributed = None

    args.world_size = 1
    args.rank = 0  # global rank

    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        # print(f'use GPU{args.device} for training')
        # print(args.world_size, args.rank)
        print('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.' % (
            args.rank, args.world_size))
    else:
        print('Training with a single process on 1 GPUs.')

    # pprint.pprint(OrderedDict(cfg.__dict__))
    if args.local_rank == 0:
        print(cfg)

    train_tsfm, valid_tsfm = get_transform(cfg)
    if args.local_rank == 0:
        print(train_tsfm)

    if cfg.DATASET.TYPE == 'multi_label':
        train_set = COCO14(cfg=cfg, split=cfg.DATASET.TRAIN_SPLIT, transform=train_tsfm,
                           target_transform=cfg.DATASET.TARGETTRANSFORM)

        valid_set = COCO14(cfg=cfg, split=cfg.DATASET.VAL_SPLIT, transform=valid_tsfm,
                           target_transform=cfg.DATASET.TARGETTRANSFORM)
    else:
        assert False, ''

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_set, shuffle=False)

    else:
        train_sampler = None
        valid_sampler = None

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.TRAIN.BATCH_SIZE // dist.get_world_size() if args.distributed else 16,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=4,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=cfg.TRAIN.BATCH_SIZE // dist.get_world_size() if args.distributed else 16,
        sampler=valid_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )

    if args.local_rank == 0:
        print('-' * 60)
        print(f'{cfg.DATASET.NAME} attr_num : {train_set.attr_num}, eval_attr_num : {train_set.eval_attr_num} '
              f'{cfg.DATASET.TRAIN_SPLIT} set: {len(train_loader.dataset)}, '
              f'{cfg.DATASET.TEST_SPLIT} set: {len(valid_loader.dataset)}, '
              )

    labels = train_set.label
    label_ratio = labels.mean(0) if cfg.LOSS.SAMPLE_WEIGHT else None

    backbone = model_dict[cfg.BACKBONE.TYPE][0]
    # backbone = TResnetL()


    # state = torch.load('./pretrained/tresnet_l.pth', map_location='cpu')
    # filtered_dict = {k: v for k, v in state['model'].items() if
    #                  (k in backbone.state_dict() and 'head.fc' not in k)}
    # backbone.load_state_dict(filtered_dict, strict=False)

    classifier = classifier_dict[cfg.CLASSIFIER.TYPE](nattr=train_set.attr_num,
                                                      c_in=model_dict[cfg.BACKBONE.TYPE][1],
                                                      # c_in=2432,
                                                      bn=cfg.CLASSIFIER.BN,
                                                      pool=cfg.CLASSIFIER.POOLING,
                                                      scale=cfg.CLASSIFIER.SCALE,
                                                      )
    model = FeatClassifier(backbone, classifier, bn_wd=cfg.TRAIN.BN_WD)
    if args.local_rank == 0:
        print(f"backbone: {cfg.BACKBONE.TYPE}, classifier: {cfg.CLASSIFIER.TYPE}")
        print(f"model_name: {cfg.NAME}")

    # with torch.cuda.device(0):
    #     model = model.cuda()
    #
    #     macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=False,
    #                                              print_per_layer_stat=False)
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    model = model.cuda()
    if args.distributed:
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)
    else:
        model = torch.nn.DataParallel(model)

    model_ema = None
    if cfg.TRAIN.EMA.ENABLE:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model, decay=cfg.TRAIN.EMA.DECAY, device='cpu' if cfg.TRAIN.EMA.FORCE_CPU else None)

    if cfg.RELOAD.TYPE:
        reload_path = os.path.join(exp_dir, cfg.RELOAD.NAME, 'img_model')
        model = get_reload_weight(reload_path, model)

    loss_weight = cfg.LOSS.LOSS_WEIGHT
    criterion = loss_dict[cfg.LOSS.TYPE](sample_weight=label_ratio)

    if cfg.TRAIN.BN_WD:
        param_groups = [{'params': model.module.finetune_params(),
                         'lr': cfg.TRAIN.LR_SCHEDULER.LR_FT,
                         'weight_decay': cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY},
                        {'params': model.module.fresh_params(),
                         'lr': cfg.TRAIN.LR_SCHEDULER.LR_NEW,
                         'weight_decay': cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY}]
    else:
        # bn parameters are not applied with weight decay
        ft_params = seperate_weight_decay(
            model.module.finetune_params(),
            lr=cfg.TRAIN.LR_SCHEDULER.LR_FT,
            weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY)

        fresh_params = seperate_weight_decay(
            model.module.fresh_params(),
            lr=cfg.TRAIN.LR_SCHEDULER.LR_NEW,
            weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY)

        param_groups = ft_params + fresh_params

    if cfg.TRAIN.OPTIMIZER.TYPE.lower() == 'sgd':
        optimizer = torch.optim.SGD(param_groups, momentum=cfg.TRAIN.OPTIMIZER.MOMENTUM)
    elif cfg.TRAIN.OPTIMIZER.TYPE.lower() == 'adam':
        optimizer = torch.optim.Adam(param_groups)
    elif cfg.TRAIN.OPTIMIZER.TYPE.lower() == 'adamw':
        optimizer = AdamW(param_groups)
    else:
        assert None, f'{cfg.TRAIN.OPTIMIZER.TYPE} is not implemented'

    if cfg.TRAIN.LR_SCHEDULER.TYPE == 'plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=4)
        if cfg.CLASSIFIER.BN:
            assert False, 'BN can not compatible with ReduceLROnPlateau'
    elif cfg.TRAIN.LR_SCHEDULER.TYPE == 'multistep':
        lr_scheduler = MultiStepLR(optimizer, milestones=cfg.TRAIN.LR_SCHEDULER.LR_STEP, gamma=0.1)
    elif cfg.TRAIN.LR_SCHEDULER.TYPE == 'warmup_cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=cfg.TRAIN.MAX_EPOCH,
            lr_min=1e-5,
            warmup_lr_init=1e-4,
            warmup_t=cfg.TRAIN.MAX_EPOCH * cfg.TRAIN.LR_SCHEDULER.WMUP_COEF,
        )
    elif cfg.TRAIN.LR_SCHEDULER.TYPE == 'onecycle':
        lr_scheduler = OneCycleLR(optimizer,
                                  max_lr=cfg.TRAIN.LR_SCHEDULER.LR_NEW,
                                  steps_per_epoch=len(train_loader),
                                  epochs=40,
                                  pct_start=0.2)

        if args.local_rank == 0:
            print(f'steps_per_epoch {len(train_loader)}')

    else:
        assert False, f'{cfg.LR_SCHEDULER.TYPE} has not been achieved yet'

    best_metric, epoch = trainer(cfg, args, epoch=cfg.TRAIN.MAX_EPOCH,
                                 model=model, model_ema=model_ema,
                                 train_loader=train_loader,
                                 valid_loader=valid_loader,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 lr_scheduler=lr_scheduler,
                                 path=save_model_path,
                                 loss_w=loss_weight,
                                 viz=visdom,
                                 tb_writer=writer)
    if args.local_rank == 0:
        print(f'{cfg.NAME},  best_metrc : {best_metric} in epoch{epoch}')


def trainer(cfg, args, epoch, model, model_ema, train_loader, valid_loader, criterion, optimizer, lr_scheduler,
            path, loss_w, viz, tb_writer):
    maximum = float(-np.inf)
    maximum_ema = float(-np.inf)
    best_epoch = 0

    result_list = defaultdict()

    for e in range(epoch):

        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        lr = optimizer.param_groups[1]['lr']

        train_loss, train_gt, train_probs, train_imgs = batch_trainer(
            cfg,
            args=args,
            epoch=e,
            model=model,
            model_ema=model_ema,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            loss_w=loss_w,
            scheduler=lr_scheduler,
        )

        if args.distributed:
            if args.local_rank == 0:
                print("Distributing BatchNorm running means and vars")
            distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

        # valid_loss, valid_gt, valid_probs, valid_imgs = valid_trainer(
        #     args=args,
        #     model=model,
        #     valid_loader=valid_loader,
        #     criterion=criterion,
        #     loss_w=loss_w
        # )

        if model_ema is not None:  # and not cfg.TRAIN.EMA.FORCE_CPU:

            if args.local_rank == 0:
                print('using model_ema to validate')

            if args.distributed:
                distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')

            valid_loss, valid_gt, valid_probs, valid_probs_ema, valid_imgs = valid_trainer(
                args=args,
                model=model,
                ema_model=model_ema.module,
                valid_loader=valid_loader,
                criterion=criterion,
                loss_w=loss_w
            )

        # if cfg.TRAIN.LR_SCHEDULER.TYPE == 'plateau':
        #     lr_scheduler.step(metrics=valid_loss)
        # elif cfg.TRAIN.LR_SCHEDULER.TYPE == 'warmup_cosine':
        #     lr_scheduler.step(epoch=e + 1)
        # else:
        #     lr_scheduler.step()

        if cfg.METRIC.TYPE == 'multi_label':

            train_metric = get_multilabel_metrics(train_gt, train_probs)
            valid_metric = get_multilabel_metrics(valid_gt, valid_probs)

            if model_ema is not None:  # and not cfg.TRAIN.EMA.FORCE_CPU:
                valid_metric_ema = get_multilabel_metrics(valid_gt, valid_probs_ema)

            if args.local_rank == 0:
                print(
                    'Performance : mAP: {:.4f}, OP: {:.4f}, OR: {:.4f}, OF1: {:.4f} CP: {:.4f}, CR: {:.4f}, '
                    'CF1: {:.4f}'.format(valid_metric.map, valid_metric.OP, valid_metric.OR, valid_metric.OF1,
                                         valid_metric.CP, valid_metric.CR, valid_metric.CF1))
                print(
                    'EMA Performance : mAP: {:.4f}, OP: {:.4f}, OR: {:.4f}, OF1: {:.4f} CP: {:.4f}, CR: {:.4f}, '
                    'CF1: {:.4f}'.format(valid_metric_ema.map, valid_metric_ema.OP, valid_metric_ema.OR,
                                         valid_metric_ema.OF1,
                                         valid_metric_ema.CP, valid_metric_ema.CR, valid_metric_ema.CF1))
                print(f'{time_str()}')
                print('-' * 60)

                tb_writer.add_scalars('train/lr', {'lr': lr}, e)

                tb_writer.add_scalars('train/loss', {'train': train_loss,
                                                     'test': valid_loss}, e)

                tb_writer.add_scalars('train/perf', {'mAP': train_metric.map,
                                                     'OP': train_metric.OP,
                                                     'OR': train_metric.OR,
                                                     'OF1': train_metric.OF1,
                                                     'CP': train_metric.CP,
                                                     'CR': train_metric.CR,
                                                     'CF1': train_metric.CF1}, e)

                tb_writer.add_scalars('test/perf', {'mAP': valid_metric.map,
                                                    'OP': valid_metric.OP,
                                                    'OR': valid_metric.OR,
                                                    'OF1': valid_metric.OF1,
                                                    'CP': valid_metric.CP,
                                                    'CR': valid_metric.CR,
                                                    'CF1': valid_metric.CF1}, e)

                tb_writer.add_scalars('test/ema_perf', {'mAP': valid_metric_ema.map,
                                                        'OP': valid_metric_ema.OP,
                                                        'OR': valid_metric_ema.OR,
                                                        'OF1': valid_metric_ema.OF1,
                                                        'CP': valid_metric_ema.CP,
                                                        'CR': valid_metric_ema.CR,
                                                        'CF1': valid_metric_ema.CF1}, e)

            cur_metric = valid_metric.map
            if cur_metric > maximum:
                maximum = cur_metric
                best_epoch = e
                save_ckpt(model, path, e, maximum)

            cur_metric = valid_metric_ema.map
            if cur_metric > maximum_ema:
                maximum_ema = cur_metric
                best_epoch = e
                save_ckpt(model, path, e, maximum_ema)


            result_list[e] = {
                'train_result': train_metric, 'valid_result': valid_metric,
                'train_gt': train_gt, 'train_probs': train_probs,
                'valid_gt': valid_gt, 'valid_probs': valid_probs
            }
        else:
            assert False, f'{cfg.METRIC.TYPE} is unavailable'

        with open(os.path.join(os.path.dirname(path), 'metric_log.pkl'), 'wb') as f:
            pickle.dump(result_list, f)

    return maximum, best_epoch


def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--cfg", help="decide which cfg to use", type=str,
        default="./configs/coco.yaml",
    )

    parser.add_argument("--debug", type=str2bool, default="true")
    parser.add_argument('--local_rank', help='node rank for distributed training', default=0,
                        type=int)
    parser.add_argument('--dist_bn', type=str, default='',
                        help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = argument_parser()

    update_config(cfg, args)

    main(cfg, args)
