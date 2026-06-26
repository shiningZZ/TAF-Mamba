import importlib
import logging

import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
from .base_model_for_mamba import BaseModel
from models.utils import get_root_logger, imwrite, tensor2img
import math
loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')
import importlib
from os import path as osp
import torch.nn as nn
from .utils import scandir
import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from collections import OrderedDict, defaultdict  # 添加 defaultdict 导入


from functools import partial
class AMPLoss(nn.Module):
    def __init__(self):
        super(AMPLoss, self).__init__()
        self.cri = nn.L1Loss()

    def forward(self, x, y):
        x = torch.fft.rfft2(x, norm='backward')
        x_mag =  torch.abs(x)
        y = torch.fft.rfft2(y, norm='backward')
        y_mag = torch.abs(y)

        return self.cri(x_mag,y_mag)


class PhaLoss(nn.Module):
    def __init__(self):
        super(PhaLoss, self).__init__()
        self.cri = nn.L1Loss()

    def forward(self, x, y):
        x = torch.fft.rfft2(x, norm='backward')
        x_mag = torch.angle(x)
        y = torch.fft.rfft2(y, norm='backward')
        y_mag = torch.angle(y)

        return self.cri(x_mag, y_mag)
def dynamic_instantiation(modules, cls_type, opt):
    """Dynamically instantiate class.

    Args:
        modules (list[importlib modules]): List of modules from importlib
            files.
        cls_type (str): Class type.
        opt (dict): Class initialization kwargs.

    Returns:
        class: Instantiated class.
    """
    print(modules,'2222222')
    print(cls_type,'8888')
    for module in modules:
        cls_ = getattr(module, cls_type, None)
        if cls_ is not None:
            break
    if cls_ is None:
        raise ValueError(f'{cls_type} is not found.')
    print(cls_)
    # exit()
    return cls_(**opt)


arch_folder = osp.dirname(osp.abspath(__file__))
arch_folder = os.path.join(arch_folder,'archs')
# arch_filenames = [
#     osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder)
#     if v.endswith('_arch.py')
# ]
#todo:修改模型import
arch_filenames = ["TAFMambawomapwofft_improve_multi_FDC4_HG_arch"]
_arch_modules = [
    importlib.import_module(f'basicsr.models.archs.{file_name}')
    for file_name in arch_filenames
]
print(_arch_modules)

def define_network(opt):
    network_type = opt.pop('type')
    print(_arch_modules, network_type, arch_filenames, arch_folder)
    # exit()
    net = dynamic_instantiation(_arch_modules, network_type, opt)
    return net


class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1,1)).item()
    
        r_index = torch.randperm(target.size(0)).to(self.device)
    
        target = lam * target + (1-lam) * target[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
    
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_

class MedicalModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(MedicalModel, self).__init__(opt)

        # define network

        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta       = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity     = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        print(load_path)
        # exit()
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
            self.cri_amp = AMPLoss().to(self.device)
            self.cri_phase = PhaLoss().to(self.device)
        else:
            raise ValueError('pixel loss are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.cond_code = data['cond_code'].to(self.device)
        self.target_mask = data['target_mask'].to(self.device)
        self.seg = data['seg'].to(self.device)
        # print(self.lq.shape,'train shape===============')
        # exit()
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.cond_code = data['cond_code'].to(self.device)  # 新增测试输入
        self.target_mask = data['target_mask'].to(self.device)
        self.seg = data['seg'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        preds = self.net_g(self.lq, self.cond_code,self.seg)
        # print(preds)
        # print(self.net_g)
        # exit()
        # exit()
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        loss_dict = OrderedDict()
        # pixel loss
        l_pix = 0.
        for pred in preds:
            # pred形状: [B, 4, H, W]（4个模态）
            # target_mask形状: [B, 4]，指示哪些模态是目标

            # 将掩码扩展到空间维度
            mask = self.target_mask.unsqueeze(2).unsqueeze(3)  # [B, 4, 1, 1]
            # print("mask",mask)
            # print(mask.shape)
            # print(self.gt.shape)
            masked_pred = pred * mask
            masked_gt = self.gt * mask

            # 计算掩码区域的损失
            l_amp = 0.05 * self.cri_amp(masked_pred, masked_gt)
            l_pha = 0.05 * self.cri_phase(masked_pred, masked_gt)
            l_p = self.cri_pix(masked_pred, masked_gt)

            # 除以掩码中1的数量进行归一化
            mask_sum = mask.sum() + 1e-8
            l_pix += (l_p + l_amp + l_pha) / mask_sum

        loss_dict['l_p'] = l_p
        loss_dict['l_pha'] = l_pha
        loss_dict['l_amp'] = l_amp

        l_pix.backward()
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):        
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        print(self.lq.shape)
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        print(img.shape,'098')
        # print(img)
        # exit()
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq      
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img, self.cond_code,self.seg)  # 传递新增的cond_code
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img, self.cond_code,self.seg)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        # 定义模态名称列表
        all_mods = ['t1c', 't1n', 't2w', 't2f']
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')


        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0
        # names = []
        combinations = []
        for i in range(5,9):  # 1-14对应二进制0001到1110
            combinations.append(i)
        print(f"combinations: {combinations}")
        all_metrics = {comb: defaultdict(list) for comb in combinations}

        for comb_code in combinations:
            dataloader.dataset.fixed_comb = comb_code
            binary = f"{comb_code:04b}"
            target_mask = [1 - int(bit) for bit in binary]  # 1表示需要保存的通道

            for idx, val_data in enumerate(dataloader):
                img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
                frame_name = val_data['lq_path'][0].split('/')[-1]
                patient_name = val_data['lq_path'][0].split('/')[-2]
                print("frame_name", frame_name)
                print('patient_name', patient_name)
                # print(val_data['lq'].shape,'valshape===========')
                # print(img_name)
                # names.append(img_name)

                self.feed_data(val_data)
                test()

                visuals = self.get_current_visuals()
                sr_img_tensor = visuals['result']  # [1, 4, H, W]
                if 'gt' in visuals:
                    gt_img_tensor = visuals['gt']  # [1, 4, H, W]
                    #gt_img = tensor2img([gt_img_tensor], rgb2bgr=rgb2bgr)

                # sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
                # if 'gt' in visuals:
                #     gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                #     del self.gt

                # tentative for out of GPU memory
                del self.lq
                del self.output
                torch.cuda.empty_cache()

                if save_img:
                    save_dir = osp.join(
                        self.opt['path']['visualization'],
                        f"{current_iter}/comb_{comb_code:04b}/{patient_name}/{frame_name}/"
                    )
                    os.makedirs(save_dir, exist_ok=True)

                    for i in range(sr_img_tensor.shape[1]):
                        if target_mask[i] == 1:  # 只保存目标通道
                            mod_name = all_mods[i]
                            # 获取单通道图像
                            sr_channel_tensor2 = sr_img_tensor[:, i:i+1, ...]
                            sr_channel_img = tensor2img([sr_channel_tensor2], rgb2bgr=rgb2bgr)

                            gt_img_tensor2 = gt_img_tensor[:, i:i + 1, ...]
                            gt_channel_img = tensor2img([gt_img_tensor2], rgb2bgr=rgb2bgr)
                            # 保存（使用模态名称）
                            imwrite(sr_channel_img, osp.join(save_dir, f'{mod_name}.png'))
                            imwrite(gt_channel_img, osp.join(save_dir, f'{mod_name}_gt.png'))


                if with_metrics:
                    # calculate metrics
                    opt_metric = deepcopy(self.opt['val']['metrics'])
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        # 计算每个目标通道的指标
                        print("sr_img_tensor",sr_img_tensor.shape)
                        print("gt_img_tensor",gt_img_tensor.shape)
                        for i in range(sr_img_tensor.shape[1]):
                            if target_mask[i] == 1:
                                mod_name = all_mods[i]
                                # 单通道指标计算
                                metric_value = getattr(metric_module, metric_type)(
                                    sr_img_tensor[:, i:i+1, ...],
                                    gt_img_tensor[:, i:i+1, ...],
                                    **opt_
                                )
                                all_metrics[comb_code][f'{name}_{mod_name}'].append(metric_value)
                    #计算平均指标
                    print(f"组合 {comb_code} ({binary}):")
                    for metric, values in all_metrics[comb_code].items():
                        if values:
                            avg_value = sum(values) / len(values)
                            print(f"  {metric}: {avg_value:.4f}")

        # 打印所有组合的平均指标
        logger = get_root_logger()
        logger.info("="*50)
        logger.info("各模态组合平均指标:")
        logger.info("="*50)
        for comb_code in combinations:
            binary = f"{comb_code:04b}"
            logger.info(f"组合 {comb_code} ({binary}):")
            for metric, values in all_metrics[comb_code].items():
                if values:
                    avg_value = sum(values) / len(values)
                    logger.info(f"  {metric}: {avg_value:.4f}")
            logger.info("-"*40)
        return all_metrics


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger,cnt = 200):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        # if cnt != 200:
            # log_str += f'\t # cnt NAN comes'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        # print(self.lq)
        # print(self.output)
        # exit()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
