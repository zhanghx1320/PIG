import os.path
import time

import torch


class PIGConfig:
    class Data:
        # 数据配置

        data_dir = os.path.join('..', 'data', 'slices_512')
        img_size = (512, 512)  # 图像resize后的大小（resize后输入至网络）
        max_pixel = 255

    class Model:
        # 模型配置

        in_channels = 1  # 输入图像通道数
        out_channels = 1  # 输出图像通道数
        ch = 32  # 基准通道数
        temb_ch = ch * 4  # 每个time_embedding的长度
        ch_mult = [1, 1, 2, 2, 4, 4]  # 提取特征时对基准通道数进行翻倍
        num_res_blocks = 2  # 每次提取特征使用的残差块数量
        initial_resolution = (512, 512)  # 图像输入网络前的大小，需要和img_size保持一致
        attn_resolutions = [(16, 16), ]  # 当图像特征大小在attn_resolutions里时加入注意力模块，长宽比需要与初始分辨率一致
        dropout = 0.0
        resamp_with_conv = True  # 为True时使用卷积进行上/下采样，否则直接插值/池化进行上/下采样

        xy_input_type_list = ['add', 'concat']
        xy_input_type = xy_input_type_list[1]  # 'add','concat'

    class Diffusion:
        # 扩散过程配置

        beta_schedule = 'linear'  # beta的取值方法:'quad','linear','const','jsd','sigmoid'
        beta_start = 1e-4
        beta_end = 2e-2
        num_diffusion_timesteps = 1024

    class Optim:
        # 优化器配置

        weight_decay = 0.000
        optimizer = 'Adam'  # 'Adam','RMSProp','SGD'
        lr = 0.0002
        beta1 = 0.9
        amsgrad = False
        eps = 0.00000001

    class LRScheduler:
        # 学习率调度器配置

        type = 'ReduceLROnPlateau'
        factor = 1 / 4
        patience = 4

    class Train:
        # 训练条件模型配置

        data_list = ['img', 'mask', 'masked', 'multi_mask', 'patient_info']
        x = data_list[1]
        y = data_list[0]
        model_type_list = ['x', 'y']
        train_model_type = model_type_list[0]
        desc = f'x_{x}_y_{y}_{train_model_type}'
        x_model_path = os.path.join('.', 'log', '20241231_170012_cond_mask_target_img_cond_exp6',
                                       'best_epoch125_loss0.3461_cond_mask_target_img_cond_exp6.pth')  # 条件模型路径，训练目标模型时用于生成条件
        binary = False  # 是否进行二值化
        binary_thre_weights = [3 / 4, 1 / 4]  # 二值化阈值=binary_thre_weights[0]*max+binary_thre_weights[1]*min
        ternary = False  # 是否进行三值化
        ternary_thre_weights = [(1 / 3, 2 / 3), (7 / 8, 1 / 8)]  # [thre0,thre1],thre_weights[1][0]>thre_weights[0][0]
        loss_weight_flag = True  # 训练目标模型时是否按cond加权
        resume_train = False  # 继续训练
        ckpt_path = os.path.join('.', 'log', '20241210_113429_cond_mask_target_img_exp1',
                                 "ckpt_epoch7_loss13.1026_cond_mask_target_img_exp1.pth")  # 继续训练使用的checkpoint路径
        log_dir = os.path.join('.', 'log',
                               f"{time.strftime('%Y%m%d_%H%M%S', time.localtime())}_{desc}")  # checkpoint保存目录
        batch_size = 8
        num_workers = 8
        n_epochs = 128
        save_freq = 8  # 保存模型的epoch频率

    class Sample:
        # 多模型采样配置

        x_model_path = os.path.join('.', 'log', 'cond_multi_mask_target_img', 'exp1',
                                         'uncond_multi_mask.pth')  # 采样使用的模型路径
        y_model_path = os.path.join('.', 'log', 'cond_multi_mask_target_img', 'exp1',
                                       'best_epoch92_loss5.7169_cond_multi_mask_target_img_exp1.pth')  # 采样使用的模型路径
        data_list = ['img', 'mask', 'masked', 'multi_mask', 'patient_info']
        x = data_list[1]  # 'mask','img'
        y = data_list[0]  # 'img','mask'
        x_binary = False  # 是否对条件进行二值化
        x_ternary = False  # 是否对条件进行三值化
        y_binary = False  # 是否对目标进行二值化
        y_ternary = False  # 是否对目标进行三值化
        binary_thre_weights = [3 / 4, 1 / 4]  # 二值化阈值=binary_thre_weights[0]*max+binary_thre_weights[1]*min
        ternary_thre_weights = [(1 / 3, 2 / 3), (7 / 8, 1 / 8)]  # [thre0,thre1],thre_weights[1][0]>thre_weights[0][0]
        desc = f'x_{x}_y_{y}_exp1_2'
        result_type = 'sequence'  # 'last':只保存最后结果,'sequence':保存去噪过程
        skip_type = 'uniform'  # 'uniform':步长均匀,'quad':平方步长
        num_sample_timesteps = 1024  # 采样的步数，在扩散的总步数//采样步数!=0时实际采样步数会大于设定值
        save_num = 16  # 当result_type == 'sequence'时等间隔保存save_num个时间步的结果
        eta = 0.0
        img_num = 64  # 生成图像的数量
        batch_size = 2
        img_size = (512, 512)  # 输出图像resize后的大小
        result_dir = os.path.join('.', 'result', f"{time.strftime('%Y%m%d_%H%M%S', time.localtime())}_{desc}")

    data = Data()
    model = Model()
    diffusion = Diffusion()
    optim = Optim()
    lr_scheduler = LRScheduler()
    train = Train()
    sample = Sample()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
