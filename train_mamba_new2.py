import argparse
import datetime
import logging
import math
import random
import time
import torch
from os import path as osp
import sys
import os

from basicsr.data import create_dataloader, create_dataset
from models import create_model

from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                           set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse

import numpy as np


def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, default="/root/autodl-tmp/code/TAFMamba-my/options/train/train.yml",
        help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()
    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])
    return opt

def init_loggers(opt):
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize wandb logger before tensorboard logger to allow proper sync:
    if (opt['logger'].get('wandb')
            is not None) and (opt['logger']['wandb'].get('project')
                              is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, (
            'should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join('tb_logger', opt['name']))
    return logger, tb_logger

def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'],
                                            opt['rank'], dataset_enlarge_ratio)
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: '
                f'{len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loader, total_epochs, total_iters

def split_patients_into_groups(patient_list, num_groups=10):
    """将病人列表平均分成num_groups组"""
    np.random.shuffle(patient_list)  # 随机打乱病人顺序
    return np.array_split(patient_list, num_groups)

def get_modality_combinations_by_stage(current_iter):
    """根据当前迭代次数返回对应的模态组合"""
    if current_iter < 10000:
        # 阶段1: 10000以下训练组合[7,11,13,14]
        return [7, 11, 13, 14]
    elif current_iter < 20000:
        # 阶段2: 10000-20000训练组合[3,5,6,7,9,10,11,12,13,14]
        return [3, 5, 6, 7, 9, 10, 11, 12, 13, 14]
    else:
        # 阶段3: 20000以上训练组合[1-14]
        return list(range(1, 15))

def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=True)

    torch.backends.cudnn.benchmark = True

    # automatic resume ..
    state_folder_path = 'experiments/{}/training_states/'.format(opt['name'])
    try:
        states = os.listdir(state_folder_path)
    except:
        states = []

    resume_state = None
    if len(states) > 0:
        max_state_file = '{}.state'.format(max([int(x[0:-6]) for x in states]))
        resume_state = os.path.join(state_folder_path, max_state_file)
        opt['path']['resume_state'] = resume_state

    # load resume states if necessary
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt[
                'name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join('tb_logger', opt['name']))

    # initialize loggers
    logger, tb_logger = init_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result

    # create model
    if resume_state:  # resume training
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
                    f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        model = create_model(opt)
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                         "Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    iters = opt['datasets']['train'].get('iters')
    batch_size = opt['datasets']['train'].get('batch_size_per_gpu')
    mini_batch_sizes = opt['datasets']['train'].get('mini_batch_sizes')
    gt_size = opt['datasets']['train'].get('gt_size')
    mini_gt_sizes = opt['datasets']['train'].get('gt_sizes')

    groups = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))])
    logger_j = [True] * len(groups)


    all_patients = sorted(os.listdir(opt['datasets']['train']['dataroot']))
    all_patients = [p for p in all_patients if os.path.isdir(os.path.join(opt['datasets']['train']['dataroot'], p))]
    num_patient_groups = 10
    patient_groups = split_patients_into_groups(all_patients, num_patient_groups)
    current_patient_group_idx = 0


    batch_per_comb = 8

    comb_total_batches = {comb: 0 for comb in range(1, 15)}

    epoch = start_epoch
    while current_iter <= total_iters:
        modality_combinations = get_modality_combinations_by_stage(current_iter)
        logger.info(f"\n===== 当前迭代次数: {current_iter}, 使用阶段: {modality_combinations} =====")

        current_patients = patient_groups[current_patient_group_idx]
        logger.info(f"使用病人组 {current_patient_group_idx + 1}/{num_patient_groups} "
                    f"（包含 {len(current_patients)} 个病人）")

        def get_group_paths(patients):
            paths = []
            for patient in patients:
                patient_dir = os.path.join(opt['datasets']['train']['dataroot'], patient)
                if not os.path.isdir(patient_dir):
                    continue
                for frame in sorted(os.listdir(patient_dir)):
                    frame_dir = os.path.join(patient_dir, frame)
                    if not os.path.isdir(frame_dir):
                        continue

                    mod_paths = {}
                    valid = True
                    for mod in ['t1c', 't1n', 't2w', 't2f']:
                        found = False
                        for ext in ['.png', '.jpg', '.tif', '.bmp', '.npy']:
                            mod_path = os.path.join(frame_dir, f"{patient}-{mod}{ext}")
                            if os.path.isfile(mod_path):
                                mod_paths[mod] = mod_path
                                found = True
                                break
                        if not found:
                            valid = False
                            break
                    if valid:
                        paths.append({'mod_paths': mod_paths, 'frame_dir': frame_dir})
            return paths


        train_loader.dataset.paths = get_group_paths(current_patients)
        logger.info(f"当前病人组包含 {len(train_loader.dataset.paths)} 帧图像")

        prefetcher.reset()
        random.shuffle(modality_combinations)


        for comb_code in modality_combinations:
            logger.info(f"训练组合 {comb_code}（二进制: {comb_code:04b}）")
            train_loader.dataset.fixed_comb = comb_code  # 设置当前组合
            train_sampler.set_epoch(epoch)  # 打乱当前病人组的帧顺序
            train_data = prefetcher.next()


            batch_count = 0
            while train_data is not None and batch_count < batch_per_comb and current_iter <= total_iters:
                current_iter += 1
                batch_count += 1
                comb_total_batches[comb_code] += 1  # 记录总训练量


                if current_iter in [10000, 20000]:
                    logger.info(f"===== 迭代次数达到 {current_iter}，切换到下一训练阶段 =====")
                    break


                model.update_learning_rate(
                    current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))

                j = ((current_iter > groups) != True).nonzero()[0]
                if len(j) == 0:
                    bs_j = len(groups) - 1
                else:
                    bs_j = j[0]
                mini_gt_size = mini_gt_sizes[bs_j]
                mini_batch_size = mini_batch_sizes[bs_j]
                if logger_j[bs_j]:
                    logger.info(
                        '\n Updating Patch_Size to {} and Batch_Size to {} \n'.format(
                            min(mini_gt_size, gt_size),
                            min(mini_batch_size, batch_size) * torch.cuda.device_count()
                        )
                    )
                    logger_j[bs_j] = False

                lq = train_data['lq']
                gt = train_data['gt']
                cond_code = train_data['cond_code']
                target_mask = train_data['target_mask']
                seg = train_data['seg']

                model.feed_train_data({'lq': lq, 'gt': gt, 'cond_code': cond_code, 'target_mask': target_mask,'lq_path':train_data['lq_path'],'seg':seg})
                model.optimize_parameters(current_iter)


                iter_time = time.time() - iter_time
                if current_iter % opt['logger']['print_freq'] == 0:
                    log_vars = {'epoch': epoch, 'iter': current_iter}
                    log_vars.update({'lrs': model.get_current_learning_rate()})
                    log_vars.update({'time': iter_time, 'data_time': data_time})
                    log_vars.update(model.get_current_log())
                    msg_logger(log_vars)
                if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    model.save(epoch, current_iter)
                if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                    logger.info("Validation...")
                    rgb2bgr = opt['val'].get('rgb2bgr', True)
                    use_image = opt['val'].get('use_image', True)
                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'], rgb2bgr, use_image)

                data_time = time.time()
                iter_time = time.time()
                train_data = prefetcher.next()


            if current_iter in [10000, 20000]:
                break

        if current_iter in [10000, 20000]:
            continue


        current_patient_group_idx = (current_patient_group_idx + 1) % num_patient_groups
        epoch += 1

    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        model.validation(val_loader, current_iter, tb_logger,
                         opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()

if __name__ == '__main__':
    main()