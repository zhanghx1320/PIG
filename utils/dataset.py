import os.path
from glob import glob

import torch
from torch import nn
from torch.utils import data
import torchvision.transforms as transforms

from utils.read_jpg import read_jpg_gray


class NormalizeTransform:
    # 将[0,max_pixel]归一化到[-1,1]
    def __init__(self, max_pixel=255):
        self.max_pixel = max_pixel

    def __call__(self, img):
        img = torch.clamp(img, 0.0, self.max_pixel)
        return img / self.max_pixel * 2 - 1


class InverseNormalizeTransform:
    # 将[-1,1]反归一化到[0,max_pixel]
    def __init__(self, max_pixel=255):
        self.max_pixel = max_pixel

    def __call__(self, img):
        img = torch.clamp(img, -1.0, 1.0)
        pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2, count_include_pad=False)
        img_pooled = pool(img)  # 去噪
        img_min = torch.min(img_pooled)
        img_max = torch.max(img_pooled)
        img = (1 + img_max) / (img_max - img_min) * img - img_max * (1 + img_min) / (
                    img_max - img_min)  # 将[min,max]线性映射到[-1,max]
        img = torch.clamp(img, -1.0, 1.0)
        return (img + 1) / 2 * self.max_pixel  # 将[-1,1]线性映射到[0,max_pixel]]


def dbtgen_transform(img_size, max_pixel=255):
    """
    将图像resize至img_size后将[0,max_pixel]归一化到[-1,1]
    Args:
        img_size: resize后的图像尺寸
        max_pixel: 最大像素值

    Returns: 图像变换

    """

    transform = transforms.Compose([transforms.Resize(img_size), NormalizeTransform(max_pixel=max_pixel)])
    return transform


def dbtgen_inverse_transform(img_size=None, max_pixel=255):
    """
    将图像resize至img_size后将[-1,1]反归一化到[0,max_pixel]
    Args:
        img_size: resize后的图像大小，为None时不resize
        max_pixel: 最大像素值

    Returns: 图像反变换

    """

    if img_size is None:
        transform = transforms.Compose([InverseNormalizeTransform(max_pixel), ])
    else:
        transform = transforms.Compose([transforms.Resize(img_size), InverseNormalizeTransform(max_pixel)])
    return transform


class DBTGenDataset(data.Dataset):
    def __init__(self, data_config):
        data_dir = data_config.data_dir
        img_size = data_config.img_size
        max_pixel = data_config.max_pixel

        self.transform = dbtgen_transform(img_size, max_pixel)

        self.img_dir = os.path.join(data_dir, 'img')
        self.mask_dir = os.path.join(data_dir, 'mask')
        self.masked_dir = os.path.join(data_dir, 'masked')
        self.multi_mask_dir = os.path.join(data_dir, 'multi_mask')

        self.img_path_list = sorted(glob(os.path.join(self.img_dir, '*.jpg')))
        self.mask_path_list = sorted(glob(os.path.join(self.mask_dir, '*.jpg')))
        self.masked_path_list = sorted(glob(os.path.join(self.masked_dir, '*.jpg')))
        self.multi_mask_path_list = sorted(glob(os.path.join(self.multi_mask_dir, '*.jpg')))

        assert len(self.img_path_list) == len(self.mask_path_list) == len(self.masked_path_list) == len(
            self.multi_mask_path_list)

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        mask_path = self.mask_path_list[index]
        masked_path = self.masked_path_list[index]
        multi_mask_path = self.multi_mask_path_list[index]

        img = torch.tensor(read_jpg_gray(img_path)).unsqueeze(0)
        mask = torch.tensor(read_jpg_gray(mask_path)).unsqueeze(0)
        masked = torch.tensor(read_jpg_gray(masked_path)).unsqueeze(0)
        multi_mask = torch.tensor(read_jpg_gray(multi_mask_path)).unsqueeze(0)
        patient_info = os.path.basename(img_path)
        img, mask, masked, multi_mask = self.transform(img), self.transform(mask), self.transform(
            masked), self.transform(multi_mask)
        return img, mask, masked, multi_mask, patient_info

    def __len__(self):
        return len(self.img_path_list)


if __name__ == '__main__':
    transform = dbtgen_inverse_transform()
    x1 = torch.randn((2, 1, 512, 512)) + 100
    x2 = torch.randn((2, 1, 512, 512)) - 100
    x = torch.concat([x1, x2], dim=0)
    x = transform(x)
    print(x[0].max(), x[0].min())
    print(x[1].max(), x[1].min())
    print(x[2].max(), x[2].min())
    print(x[3].max(), x[3].min())
