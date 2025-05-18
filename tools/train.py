import argparse
import datetime
import os
import os.path as osp
import shutil
import time

import torch
import yaml
from munch import Munch
from softgroup.data import build_dataloader, build_dataset
from softgroup.evaluation import (PanopticEval, ScanNetEval, evaluate_offset_mae,
                                  evaluate_semantic_acc, evaluate_semantic_miou)
from softgroup.model import SoftGroup
from softgroup.util import (AverageMeter, SummaryWriter, build_optimizer, checkpoint_save,
                            collect_results_cpu, cosine_lr_after_step, get_dist_info,
                            get_max_memory, get_root_logger, init_dist, is_main_process,
                            is_multiple, is_power2, load_checkpoint)
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
import numpy as np


def get_args():
    parser = argparse.ArgumentParser('SoftGroup')
    parser.add_argument('config', type=str, help='path to config file') #命令行中体现为必须是传入的第一个参数
    # parser.add_argument('--config', type=str, default='configs/softgroup/softgroup_ps_backbone.yaml',help='path to config file') #命令行中体现为必须是传入的第一个参数
    parser.add_argument('--dist', action='store_true', help='run with distributed parallel')
    parser.add_argument('--resume', type=str, help='path to resume from')
    parser.add_argument('--work_dir', type=str, help='working directory')
    parser.add_argument('--skip_validate', action='store_true', help='skip validation')
    args = parser.parse_args()
    return args


def train(epoch, model, optimizer, scaler, train_loader, cfg, logger, writer):
    model.train()
    iter_time = AverageMeter(True)
    data_time = AverageMeter(True)
    meter_dict = {}
    end = time.time()

    if train_loader.sampler is not None and cfg.dist:
        train_loader.sampler.set_epoch(epoch)

    for i, batch in enumerate(train_loader, start=1):
        data_time.update(time.time() - end)
        cosine_lr_after_step(optimizer, cfg.optimizer.lr, epoch - 1, cfg.step_epoch, cfg.epochs)
        with torch.cuda.amp.autocast(enabled=cfg.fp16):
            loss, log_vars = model(batch, return_loss=True) # batch是一个字典，包含了所有的输入数据(scan_id, coord, coord_float, feat, semantic_label, instance_label, inst_num, inst_pointnum, inst_cls, pt_offset_label)

        # meter_dict
        for k, v in log_vars.items():
            if k not in meter_dict.keys():
                meter_dict[k] = AverageMeter()
            meter_dict[k].update(v)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if cfg.get('clip_grad_norm', None):
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        # time and print
        remain_iter = len(train_loader) * (cfg.epochs - epoch + 1) - i
        iter_time.update(time.time() - end)
        end = time.time()
        remain_time = remain_iter * iter_time.avg
        remain_time = str(datetime.timedelta(seconds=int(remain_time)))
        lr = optimizer.param_groups[0]['lr']

        if is_multiple(i, 10):
            log_str = f'Epoch [{epoch}/{cfg.epochs}][{i}/{len(train_loader)}]  '
            log_str += f'lr: {lr:.2g}, eta: {remain_time}, mem: {get_max_memory()}, '\
                f'data_time: {data_time.val:.2f}, iter_time: {iter_time.val:.2f}'
            for k, v in meter_dict.items():
                log_str += f', {k}: {v.val:.4f}'
            logger.info(log_str)
    writer.add_scalar('train/learning_rate', lr, epoch)
    for k, v in meter_dict.items():
        writer.add_scalar(f'train/{k}', v.avg, epoch)
    checkpoint_save(epoch, model, optimizer, cfg.work_dir, cfg.save_freq)


def validate(epoch, model, val_loader, cfg, logger, writer):
    logger.info('Validation')
    results = []
    all_sem_preds, all_sem_labels, all_offset_preds, all_offset_labels = [], [], [], []
    all_inst_labels, all_pred_insts, all_gt_insts = [], [], []
    all_panoptic_preds = []
    _, world_size = get_dist_info() # 单一GPU时，world_size=1
    progress_bar = tqdm(total=len(val_loader) * world_size, disable=not is_main_process())
    val_set = val_loader.dataset
    eval_tasks = cfg.model.test_cfg.eval_tasks
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(val_loader):
            result = model(batch)
            results.append(result)
            progress_bar.update(world_size)
        progress_bar.close()
        results = collect_results_cpu(results, len(val_set))
    if is_main_process():
        for res in results:
            if 'semantic' in eval_tasks or 'panoptic' in eval_tasks:
                all_sem_labels.append(res['semantic_labels'])
                all_inst_labels.append(res['instance_labels'])
            if 'semantic' in eval_tasks:
                all_sem_preds.append(res['semantic_preds'])
                all_offset_preds.append(res['offset_preds'])
                all_offset_labels.append(res['offset_labels'])
            if 'instance' in eval_tasks:
                all_pred_insts.append(res['pred_instances'])
                all_gt_insts.append(res['gt_instances'])
            if 'panoptic' in eval_tasks:
                all_panoptic_preds.append(res['panoptic_preds'])
        
        if 'instance' in eval_tasks:
            logger.info('Evaluate instance segmentation')
            eval_min_npoint = getattr(cfg, 'eval_min_npoint', None)
            scannet_eval = ScanNetEval(val_set.CLASSES, eval_min_npoint)
            eval_res = scannet_eval.evaluate(all_pred_insts, all_gt_insts)
            writer.add_scalar('val/AP', eval_res['all_ap'], epoch)
            writer.add_scalar('val/AP_50', eval_res['all_ap_50%'], epoch)
            writer.add_scalar('val/AP_25', eval_res['all_ap_25%'], epoch)
            # 添加更多指标记录
            writer.add_scalar('val/AR', eval_res['all_rc'], epoch)
            writer.add_scalar('val/F1', eval_res.get('all_f1', 0), epoch)
            writer.add_scalar('val/MCov', eval_res.get('MCov', 0), epoch)
            writer.add_scalar('val/MWCov', eval_res.get('MWCov', 0), epoch)
            
            # 日志输出更多指标
            # logger.info('Leaf Instance Segmentation:')
            logger.info('AP: {:.3f}. AP_50: {:.3f}. AP_25: {:.3f}'.format(
                eval_res['all_ap'], eval_res['all_ap_50%'], eval_res['all_ap_25%']))
            logger.info('AR: {:.3f}. F1: {:.3f}'.format(
                eval_res['all_rc'], eval_res.get('all_f1', 0)))
            logger.info('MCov: {:.3f}. MWCov: {:.3f}'.format(
                eval_res.get('MCov', 0), eval_res.get('MWCov', 0)))
            
        if 'panoptic' in eval_tasks:
            logger.info('Evaluate panoptic segmentation')
            eval_min_npoint = getattr(cfg, 'eval_min_npoint', None)
            panoptic_eval = PanopticEval(val_set.THING, val_set.STUFF, min_points=eval_min_npoint)
            eval_res = panoptic_eval.evaluate(all_panoptic_preds, all_sem_labels, all_inst_labels)
            writer.add_scalar('val/PQ', eval_res[0], epoch)
            logger.info('PQ: {:.1f}'.format(eval_res[0]))
            
        # 语义分割评估，添加精度和召回率
        if 'semantic' in eval_tasks:
            logger.info('Evaluate semantic segmentation and offset MAE')
            ignore_label = cfg.model.ignore_label
            # 计算mIoU
            miou = evaluate_semantic_miou(all_sem_preds, all_sem_labels, ignore_label, logger)
            # 计算准确率
            acc = evaluate_semantic_acc(all_sem_preds, all_sem_labels, ignore_label, logger)
            # 计算偏移MAE
            mae = evaluate_offset_mae(all_offset_preds, all_offset_labels, all_inst_labels, ignore_label, logger)
            
            # 计算语义分割的精度和召回率
            # 将所有预测和标签合并为一维数组
            all_preds = np.concatenate([pred.reshape(-1) for pred in all_sem_preds])
            all_labels = np.concatenate([label.reshape(-1) for label in all_sem_labels])
            
            # 忽略标签
            mask = (all_labels != ignore_label)
            all_preds = all_preds[mask]
            all_labels = all_labels[mask]
            
            # 获取类别数量
            num_classes = cfg.model.semantic_classes
            
            # 类别精度和召回率
            precision_per_class = np.zeros(num_classes)
            recall_per_class = np.zeros(num_classes)
            f1_per_class = np.zeros(num_classes)
            iou_per_class = np.zeros(num_classes)
            
            # 对每个类别计算精度和召回率
            for i in range(num_classes):
                pred_i = (all_preds == i)
                gt_i = (all_labels == i)
                tp = np.sum(pred_i & gt_i)
                fp = np.sum(pred_i & ~gt_i)
                fn = np.sum(~pred_i & gt_i)
                
                # 避免除以零
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
                
                precision_per_class[i] = precision
                recall_per_class[i] = recall
                f1_per_class[i] = f1
                iou_per_class[i] = iou
            
            # 计算平均值
            mean_precision = np.mean(precision_per_class)
            mean_recall = np.mean(recall_per_class)
            mean_f1 = np.mean(f1_per_class)
            
            # 记录TensorBoard
            writer.add_scalar('val/mIoU', miou, epoch)
            writer.add_scalar('val/Acc', acc, epoch)
            writer.add_scalar('val/Offset_MAE', mae, epoch)
            writer.add_scalar('val/mPrecision', mean_precision, epoch)
            writer.add_scalar('val/mRecall', mean_recall, epoch)
            writer.add_scalar('val/mF1', mean_f1, epoch)
            
            # 输出日志
            logger.info('Semantic mIoU: {:.1f}'.format(miou))
            logger.info('Semantic Acc: {:.1f}'.format(acc))
            logger.info('Semantic mPrecision: {:.1f}'.format(mean_precision * 100))
            logger.info('Semantic mRecall: {:.1f}'.format(mean_recall * 100))
            logger.info('Semantic mF1: {:.1f}'.format(mean_f1 * 100))
            logger.info('Offset MAE: {:.3f}'.format(mae))
            
            # 输出每个类别的指标
            for i in range(num_classes):
                logger.info('Class {}: IoU: {:.1f}, Precision: {:.1f}, Recall: {:.1f}, F1: {:.1f}'.format(
                    i, iou_per_class[i] * 100, precision_per_class[i] * 100, recall_per_class[i] * 100, f1_per_class[i] * 100))


def main():
    args = get_args()
    cfg_txt = open(args.config, 'r').read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))

    if args.dist:
        init_dist()
    cfg.dist = args.dist

    # work_dir & logger
    if args.work_dir:
        cfg.work_dir = args.work_dir
    elif not cfg.work_dir:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    else:
        pass
    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)   # './work_dirs/softgroup_s3dis_bacbone_fold5'
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file)
    
    # 自动检测数据是否有颜色特征（RGB），并设置use_color
    if 'use_color' not in cfg.model:
        # 尝试根据数据自动推断
        # 这里假设train_set的feat shape为[N, C]，C=3为RGB，否则为无RGB
        # 先构建数据集
        train_set = build_dataset(cfg.data.train, logger)
        sample = train_set[0]
        feat_dim = sample['feat'].shape[1] if 'feat' in sample and len(sample['feat'].shape) == 2 else 0
        cfg.model.use_color = (feat_dim == 3)
        logger.info(f"'use_color' not specified in config, auto detected: {cfg.model.use_color}")
    else:
        logger.info(f"'use_color' specified in config: {cfg.model.use_color}")

    logger.info(f'Config:\n{cfg_txt}')
    logger.info(f'Distributed: {args.dist}')
    logger.info(f'Mix precision training: {cfg.fp16}')
    shutil.copy(args.config, osp.join(cfg.work_dir, osp.basename(args.config)))
    writer = SummaryWriter(cfg.work_dir)

    # model
    model = SoftGroup(**cfg.model).cuda()
    if args.dist:
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)

    # data
    # 如果train_set已提前构建，则复用，否则重新构建
    if 'train_set' not in locals():
        train_set = build_dataset(cfg.data.train, logger)
    val_set = build_dataset(cfg.data.test, logger)
    train_loader = build_dataloader(
        train_set, training=True, dist=args.dist, **cfg.dataloader.train)
    val_loader = build_dataloader(val_set, training=False, dist=args.dist, **cfg.dataloader.test)

    # optim
    optimizer = build_optimizer(model, cfg.optimizer)

    # pretrain, resume
    # load_checkpoint函数会自动覆盖传入的model和optimizer，只return epoch
    start_epoch = 1
    if args.resume:
        logger.info(f'Resume from {args.resume}')
        start_epoch = load_checkpoint(args.resume, logger, model, optimizer=optimizer)
        
    elif cfg.pretrain:
        # 如果是backbone预训练，需要在这里加载hais的预训练模型: hais_ckpt_spconv2.pth
        # 如果是整个模型训练，需要在这里加载backbone的预训练模型: *backbone/latest.pth
        logger.info(f'Load pretrain from {cfg.pretrain}')
        # TODO：如果是加载hais，为什么model还是Softgroup呢？能对上嘛？
        load_checkpoint(cfg.pretrain, logger, model)

    # train and val
    logger.info('Training')
    for epoch in range(start_epoch, cfg.epochs + 1):
        train(epoch, model, optimizer, scaler, train_loader, cfg, logger, writer)
        if not args.skip_validate and (is_multiple(epoch, cfg.save_freq) or is_power2(epoch)):
            validate(epoch, model, val_loader, cfg, logger, writer)
        writer.flush()


if __name__ == '__main__':
    main()
