from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_folder_test1200,
                                    paired_paths_from_folder_test1200_test,
                                    paired_DP_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP
from PIL import Image
import random
import numpy as np
import torch
import cv2
import os
from models.utils import get_root_logger, imwrite, tensor2img


# 生成所有可能的模态组合（4位二进制，排除0000和1111）
def generate_modality_combinations():
    combinations = []
    for i in range(1, 15):  # 1-14对应二进制0001到1110
        combinations.append(i)
    return combinations


class Dataset_MedicalImageMultiseg(data.Dataset):

    def __init__(self, opt, fixed_comb=None):
        super(Dataset_MedicalImageMultiseg, self).__init__()
        self.opt = opt

        self.file_client = None
        self.io_backend_opt = opt['io_backend']


        self.dataroot = opt['dataroot']
        self.crop_pixels = opt.get('crop_pixels', 8)  # 每边裁剪的像素数
        self.all_mods = ['t1c', 't1n', 't2w', 't2f']  # 所有4个模态
        self.mod_order = self.all_mods  # 模态顺序固定，用于二进制编码
        self.seg_suffix = opt.get('seg_suffix', '_seg')  # 分割图后缀


        self.combinations = generate_modality_combinations()


        self.cond_mean = opt.get('cond_mean', None)
        self.cond_std = opt.get('cond_std', None)
        self.tgt_mean = opt.get('tgt_mean', None)
        self.tgt_std = opt.get('tgt_std', None)


        self.__paths = self.get_paths()



        if self.opt['phase'] == 'train':
            self.geometric_augs = opt.get('geometric_augs', False)

        self.fixed_comb = fixed_comb

    def get_paths(self):

        paths = []
        for patient in sorted(os.listdir(self.dataroot)):
            patient_dir = os.path.join(self.dataroot, patient)
            if not os.path.isdir(patient_dir):
                continue

            for frame in sorted(os.listdir(patient_dir)):
                frame_dir = os.path.join(patient_dir, frame)
                if not os.path.isdir(frame_dir):
                    continue


                mod_paths = {}
                valid = True
                for mod in self.all_mods:

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


                seg_path = None
                for ext in ['.png', '.jpg']:
                    seg_candidate = os.path.join(frame_dir, f"{patient}-seg{ext}")
                    if os.path.isfile(seg_candidate):
                        seg_path = seg_candidate
                        # print(f"seg_path: {seg_path}")
                        break

                if seg_path:
                #     valid = False
                # # print(f"valid: {valid}")
                # if valid:
                    paths.append({
                        'mod_paths': mod_paths,  # 所有模态的路径
                        'frame_dir': frame_dir,
                        'seg_path': seg_path  # 分割图路径
                    })
                    # print("path seg_path", paths[-1]['seg_path'])
        return paths

    def fixed_crop(self, img):
        h, w = img.shape[:2]
        top = self.crop_pixels
        bottom = h - self.crop_pixels
        left = self.crop_pixels
        right = w - self.crop_pixels
        return img[top:bottom, left:right]

    def load_segmentation_map(self, seg_path, target_shape):
        if seg_path is None:

            h, w = target_shape[:2]
            return np.zeros((h, w, 1), dtype=np.float32)

        try:
            img_bytes = self.file_client.get(seg_path, 'gt')
            seg_img = imfrombytes(img_bytes, float32=True)


            if len(seg_img.shape) == 3 and seg_img.shape[2] > 1:
                seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
                seg_img = np.expand_dims(seg_img, axis=2)
            elif len(seg_img.shape) == 2:
                seg_img = np.expand_dims(seg_img, axis=2)


            if seg_img.max() > 1:
                seg_img = (seg_img > 0.5 * seg_img.max()).astype(np.float32)
            else:
                seg_img = seg_img.astype(np.float32)

            return seg_img
        except Exception as e:
            logger = get_root_logger()
            logger.warning(f"Failed to load segmentation map {seg_path}: {e}")
            h, w = target_shape[:2]
            return np.zeros((h, w, 1), dtype=np.float32)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)


        data_dict = self.__paths[index]
        # print(f"data_dict: {data_dict}")
        mod_paths = data_dict['mod_paths']
        frame_dir = data_dict['frame_dir']
        seg_path = data_dict['seg_path']


        if self.opt['phase'] == 'train':
            if self.fixed_comb is not None:
                comb_code = self.fixed_comb
            else:
                comb_code = np.random.choice(self.combinations)
        else:
            comb_code = self.fixed_comb


        binary = f"{comb_code:04b}"
        input_mask = [int(bit) for bit in binary]
        target_mask = [1 - bit for bit in input_mask]


        input_mods = [self.mod_order[i] for i, bit in enumerate(input_mask) if bit == 1]
        target_mods = [self.mod_order[i] for i, bit in enumerate(input_mask) if bit == 0]
        num_input = len(input_mods)
        num_target = len(target_mods)


        all_modalities = {}
        for mod in self.all_mods:
            img_bytes = self.file_client.get(mod_paths[mod], 'gt')
            img = imfrombytes(img_bytes, float32=True)
            if img.shape[2] > 1:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.expand_dims(img, axis=2)
            all_modalities[mod] = img


        reference_shape = all_modalities[self.all_mods[0]].shape
        seg_img = self.load_segmentation_map(seg_path, reference_shape)


        h, w = all_modalities[self.all_mods[0]].shape[:2]
        img_target = np.zeros((h, w, 4), dtype=np.float32)


        for i, mod in enumerate(self.all_mods):
            if target_mask[i] == 1:
                img_target[:, :, i] = all_modalities[mod][:, :, 0]


        input_imgs = []
        for mod in input_mods:
            img_bytes = self.file_client.get(mod_paths[mod], 'lq')
            img = imfrombytes(img_bytes, float32=True)

            if img.shape[2] > 1:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.expand_dims(img, axis=2)  # 保持HWC格式
            input_imgs.append(img)


        if input_imgs:
            img_input = np.concatenate(input_imgs, axis=2)
        else:
            raise ValueError("No input modalities selected")


        img_input = self.fixed_crop(img_input)
        img_target = self.fixed_crop(img_target)
        seg_img = self.fixed_crop(seg_img)


        if self.opt['phase'] == 'train' and self.geometric_augs:
            img_input, img_target, seg_img = augment([img_input, img_target, seg_img],
                                                     hflip=True, rotation=True)


        img_input = torch.from_numpy(np.ascontiguousarray(img_input.transpose(2, 0, 1)))
        img_target = torch.from_numpy(np.ascontiguousarray(img_target.transpose(2, 0, 1)))
        seg = torch.from_numpy(np.ascontiguousarray(seg_img.transpose(2, 0, 1)))


        if self.cond_mean is not None:
            cond_mean_tensor = torch.tensor(self.cond_mean).view(num_input, 1, 1)
            img_input = (img_input - cond_mean_tensor)
        if self.cond_std is not None:
            cond_std_tensor = torch.tensor(self.cond_std).view(num_input, 1, 1)
            img_input = img_input / cond_std_tensor


        if self.tgt_mean is not None:

            tgt_mean_tensor = torch.tensor(self.tgt_mean).view(num_target, 1, 1)
            img_target = (img_target - tgt_mean_tensor)
        if self.tgt_std is not None:
            tgt_std_tensor = torch.tensor(self.tgt_std).view(num_target, 1, 1)
            img_target = img_target / tgt_std_tensor

        input_mask = torch.tensor(input_mask, dtype=torch.float32)
        target_mask = torch.tensor(target_mask, dtype=torch.float32)


        return {
            'lq': img_input,
            'gt': img_target,
            'cond_code': comb_code,
            'lq_path': frame_dir,
            'input_mask': input_mask,
            'target_mask': target_mask,
            'seg': seg,
        }

    def __len__(self):
        return len(self.__paths)