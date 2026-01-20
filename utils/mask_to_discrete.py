import torch
# from matplotlib import pyplot as plt
from torch import nn



def mask_to_binary(mask, thre_weights, mask_0=-1, mask_1=1):
    """
    将mask二值化
    Args:
        mask: 图像数组
        thre_weights: 二值化阈值=thre_weights[0]*max+thre_weights[1]*min
        mask_0: 小于阈值的部分设置为mask_0
        mask_1: 大于阈值的部分设置为mask_1

    Returns:　二值化后的数组

    """
    mask = torch.clamp(mask, mask_0, mask_1)
    pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2, count_include_pad=False)
    mask = pool(mask)
    max_mask = torch.tensor([mask[i].max() for i in range(mask.shape[0])])
    min_mask = torch.tensor([mask[i].min() for i in range(mask.shape[0])])
    thre = (thre_weights[0] * max_mask + thre_weights[1] * min_mask).view(-1, 1, 1, 1).to(mask.device)
    mask = mask_0 * (mask <= thre) + mask_1 * (mask > thre)
    return mask


def mask_to_ternary(mask, thre_weights, mask_0=-1, mask_1=0, mask_2=1):
    """
    将mask三值化
    Args:
        mask: 图像数组
        thre_weights: 需满足thre_weights[0]>thre_weights[1],阈值0=thre_weights[0]*min+thre_weights[1]*max,阈值1=thre_weights[0]*max+thre_weights[1]*min
        mask_0: 小于阈值0的部分设置为mask_0
        mask_1: 大于阈值0小于阈值1的部分设置为mask_1
        mask_2: 大于阈值1的部分设置为mask_2

    Returns:　三值化后的数组

    """
    mask = torch.clamp(mask, mask_0, mask_2)
    pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2, count_include_pad=False)
    mask = pool(mask)
    max_mask = torch.tensor([mask[i].max() for i in range(mask.shape[0])])
    min_mask = torch.tensor([mask[i].min() for i in range(mask.shape[0])])
    thre_0 = (thre_weights[0][0] * max_mask + thre_weights[0][1] * min_mask).view(-1, 1, 1, 1).to(mask.device)
    thre_1 = (thre_weights[1][0] * max_mask + thre_weights[1][1] * min_mask).view(-1, 1, 1, 1).to(mask.device)
    mask = mask_0 * (mask < thre_0) + mask_1 * ((mask >= thre_0) & (mask <= thre_1)) + mask_2 * (mask > thre_1)
    return mask
