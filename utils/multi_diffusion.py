import os.path
import time

import cv2
import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.utils import data
from tqdm import tqdm

from PIG.models.multi_diffusion_model import MultiDiffusionModel
from PIG.utils.dataset import DBTGenDataset, dbtgen_inverse_transform
from PIG.utils.get_betas import get_betas
from PIG.utils.get_optimizer import get_optimizer
from PIG.utils.loss import noise_estimation_loss
from PIG.utils.mask_to_discrete import mask_to_binary, mask_to_ternary
from PIG.utils.multi_denoise import multi_denoise, one_step_denoise


class MultiDiffusion():
    def __init__(self, multi_diffusion_config):
        self.data_config = multi_diffusion_config.data
        self.model_config = multi_diffusion_config.model
        self.diffusion_config = multi_diffusion_config.diffusion
        self.optim_config = multi_diffusion_config.optim
        self.lr_scheduler_config = multi_diffusion_config.lr_scheduler
        self.train_config = multi_diffusion_config.train
        self.sample_config = multi_diffusion_config.sample
        self.device = multi_diffusion_config.device
        self.betas = get_betas(self.diffusion_config).to(self.device)
        self.num_diffusion_timesteps = self.diffusion_config.num_diffusion_timesteps

    def train(self):
        """
        训练模型
        """

        data_config = self.data_config
        model_config = self.model_config
        optim_config = self.optim_config
        lr_scheduler_config = self.lr_scheduler_config
        train_config = self.train_config
        device = self.device
        betas = self.betas
        num_diffusion_timesteps = self.num_diffusion_timesteps
        train_model_type = train_config.train_model_type
        desc = train_config.desc

        dataset = DBTGenDataset(data_config)
        train_dataloader = data.DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True,
                                           num_workers=train_config.num_workers)
        model = MultiDiffusionModel(model_config).to(device)
        model = torch.nn.DataParallel(model)
        optimizer = get_optimizer(optim_config, model.parameters())
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_scheduler_config.factor,
                                                   patience=lr_scheduler_config.patience,
                                                   verbose=True) if lr_scheduler_config.type == 'ReduceLROnPlateau' else None

        start_epoch = 0
        if train_config.resume_train:
            y_states = torch.load(train_config.ckpt_path)
            model.load_state_dict(y_states[0])
            y_states[1]["param_groups"][0]["eps"] = optim_config.eps
            optimizer.load_state_dict(y_states[1])
            start_epoch = y_states[2]

        log_dir = train_config.log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir, 'log.csv')
        if train_model_type == 'y':
            x_model_path = train_config.x_model_path
            x_model = MultiDiffusionModel(model_config).to(device)
            x_model = torch.nn.DataParallel(x_model)
            x_states = torch.load(x_model_path)
            x_model.load_state_dict(x_states[0])
            x_model.eval()
            torch.save(x_states, os.path.join(log_dir, f"x_{train_config.x}_y_{train_config.y}_x.pth"))

        with open(log_path, 'w') as f:
            f.write('epoch,lr,loss,time(s)\n')
            for epoch in range(start_epoch, start_epoch + train_config.n_epochs):
                epoch_loss = 0
                epoch_start_time = time.time()
                for i, data_list in tqdm(enumerate(train_dataloader),
                                         total=len(train_dataloader),
                                         desc=f'training_epoch_{epoch}/{start_epoch + train_config.n_epochs}'):
                    x0 = data_list[train_config.data_list.index(train_config.x)]
                    x0_num = x0.shape[0]
                    y0 = data_list[train_config.data_list.index(train_config.y)]
                    model.train()

                    x0 = x0.to(device)
                    y0 = y0.to(device)
                    e_x = torch.randn_like(x0)
                    e_y = torch.randn_like(y0)

                    t = torch.randint(low=0, high=num_diffusion_timesteps, size=(x0_num // 2 + 1,)).to(device)
                    t = torch.cat([t, num_diffusion_timesteps - t - 1], dim=0)[:x0_num]
                    a = (1 - betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
                    yt = y0 * a.sqrt() + e_y * (1.0 - a).sqrt()
                    if train_model_type == 'x':
                        loss = noise_estimation_loss(model, x0, yt, t, e_x, betas)
                    elif train_model_type == 'y':
                        loss_weight_flag = train_config.loss_weight_flag
                        xt = x0 * a.sqrt() + e_x * (1.0 - a).sqrt()
                        x0_pred = one_step_denoise(x_model, xt, yt, t, betas)
                        if train_config.binary is True:
                            thre_weights = train_config.binary_thre_weights
                            x0_pred = mask_to_binary(x0_pred, thre_weights).to(dtype=torch.float)
                        elif train_config.ternary is True:
                            thre_weights = train_config.ternary_thre_weights
                            x0_pred = mask_to_ternary(x0_pred, thre_weights).to(dtype=torch.float)
                        loss = noise_estimation_loss(model, y0, x0_pred, t, e_y, betas, loss_weight_flag=loss_weight_flag)

                    epoch_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                epoch_time = time.time() - epoch_start_time
                print(
                    f'epoch:{epoch},lr:{optimizer.param_groups[0]["lr"]},loss:{epoch_loss},time:{epoch_time}')
                if scheduler is not None:
                    scheduler.step(epoch_loss)

                # 保存检查点
                f.write(f'{epoch},{optimizer.param_groups[0]["lr"]},{epoch_loss},{epoch_time}\n')
                model_states = [model.state_dict(), optimizer.state_dict(), epoch]
                train_epoch = epoch - start_epoch  # 训练的epoch数
                if train_epoch % train_config.save_freq == 0 or epoch_loss <= min_loss:
                    # 在每save_freq个epoch的最开始初始化最优模型
                    best_states = model_states
                    min_loss = epoch_loss
                if train_epoch == 0:
                    # 保存第一个epoch的模型
                    torch.save(model_states,
                               os.path.join(log_dir, f"ckpt_epoch{epoch}_loss{epoch_loss:.4f}_{desc}.pth"))
                elif train_epoch == train_config.n_epochs - 1 or (
                        train_epoch + 1) % train_config.save_freq == 0:
                    # 每save_freq个epoch的最后保存模型和最优模型
                    torch.save(model_states,
                               os.path.join(log_dir, f"ckpt_epoch{epoch}_loss{epoch_loss:.4f}_{desc}.pth"))
                    torch.save(best_states,
                               os.path.join(log_dir, f"best_epoch{best_states[2]}_loss{min_loss:.4f}_{desc}.pth"))

    def sample(self):
        """
        多模型采样
        """

        model_config = self.model_config
        sample_config = self.sample_config
        device = self.device
        img_num = sample_config.img_num
        num_diffusion_timesteps = self.num_diffusion_timesteps
        betas = self.betas
        eta = sample_config.eta
        assert sample_config.skip_type == 'uniform' or sample_config.skip_type == 'quad', print(
            sample_config.skip_type)
        assert sample_config.result_type == 'last' or sample_config.result_type == 'sequence', print(
            sample_config.result_type)
        if sample_config.skip_type == 'uniform':
            skip = num_diffusion_timesteps // sample_config.num_sample_timesteps
            denoise_t_list = range(0, num_diffusion_timesteps, skip)
        elif sample_config.multi_sample.skip_type == "quad":
            denoise_t_list = (np.linspace(0, np.sqrt(num_diffusion_timesteps * 0.8),
                                          sample_config.num_sample_timesteps) ** 2)
            denoise_t_list = [int(s) for s in list(denoise_t_list)]
        else:
            raise NotImplementedError("Skip type not defined")

        x_model = MultiDiffusionModel(model_config)
        x_model_path = sample_config.x_model_path
        x_states = torch.load(x_model_path)
        x_model = x_model.to(device)
        x_model = torch.nn.DataParallel(x_model)
        x_model.load_state_dict(x_states[0], strict=True)
        y_model = MultiDiffusionModel(model_config)
        y_model_path = sample_config.y_model_path
        y_states = torch.load(y_model_path)
        y_model = y_model.to(device)
        y_model = torch.nn.DataParallel(y_model)
        y_model.load_state_dict(y_states[0], strict=True)
        x_model.eval()
        y_model.eval()

        with torch.no_grad():
            batch_size = sample_config.batch_size
            batch_num = np.ceil(img_num / batch_size).astype(int)
            result_dir = sample_config.result_dir
            for batch in range(batch_num):
                n_batch = min(batch_size, img_num - batch * batch_size)
                xt = torch.randn((n_batch, model_config.in_channels) + model_config.initial_resolution).to(device)
                yt = torch.randn((n_batch, model_config.in_channels) + model_config.initial_resolution).to(device)
                xts, yts, x0s, y0s = multi_denoise(x_model, y_model, xt, yt, denoise_t_list, betas, eta,
                                                   x_binary=sample_config.x_binary, x_ternary=sample_config.x_ternary,
                                                   binary_thre_weights=sample_config.binary_thre_weights,
                                                   ternary_thre_weights=sample_config.ternary_thre_weights)
                if sample_config.result_type == 'last':
                    # 只保存最后结果
                    x0, y0 = x0s[-1], y0s[-1]
                    if not os.path.exists(result_dir):
                        os.makedirs(result_dir)
                    if sample_config.x_binary is True:
                        thre_weights = sample_config.binary_thre_weights
                        x0_binary = mask_to_binary(x0.to(device), thre_weights).to(dtype=torch.float)
                        x0_binary = dbtgen_inverse_transform(img_size=sample_config.img_size)(x0_binary)
                    elif sample_config.x_ternary is True:
                        thre_weights = sample_config.ternary_thre_weights
                        x0_ternary = mask_to_ternary(x0.to(device), thre_weights).to(dtype=torch.float)
                        x0_ternary = dbtgen_inverse_transform(img_size=sample_config.img_size)(x0_ternary)
                    if sample_config.y_binary is True:
                        thre_weights = sample_config.binary_thre_weights
                        y0_binary = mask_to_binary(y0.to(device), thre_weights).to(dtype=torch.float)
                        y0_binary = dbtgen_inverse_transform(img_size=sample_config.img_size)(y0_binary)
                    if sample_config.y_ternary is True:
                        thre_weights = sample_config.ternary_thre_weights
                        y0_ternary = mask_to_ternary(y0.to(device), thre_weights).to(dtype=torch.float)
                        y0_ternary = dbtgen_inverse_transform(img_size=sample_config.img_size)(y0_ternary)
                    x0 = dbtgen_inverse_transform(img_size=sample_config.img_size)(x0)
                    y0 = dbtgen_inverse_transform(img_size=sample_config.img_size)(y0)
                    for i in tqdm(range(x0.shape[0]), desc='saving'):
                        x_path = os.path.join(result_dir,
                                              f"x0_{batch * batch_size + i}_{sample_config.x}.jpg")
                        cv2.imwrite(x_path, np.array(x0[i].cpu()).squeeze())
                        y_path = os.path.join(result_dir,
                                                   f"x0_{batch * batch_size + i}_{sample_config.y}.jpg")
                        cv2.imwrite(y_path, np.array(y0[i].cpu()).squeeze())
                        if sample_config.x_binary is True:
                            x0_binary_path = os.path.join(result_dir,
                                                          f"x0_{batch * batch_size + i}_{sample_config.x}_binary.jpg")
                            cv2.imwrite(x0_binary_path, np.array(x0_binary[i].cpu()).squeeze())
                        elif sample_config.x_ternary is True:
                            x0_ternary_path = os.path.join(result_dir,
                                                           f"x0_{batch * batch_size + i}_{sample_config.x}_ternary.jpg")
                            cv2.imwrite(x0_ternary_path, np.array(x0_ternary[i].cpu()).squeeze())
                        if sample_config.y_binary is True:
                            y0_binary_path = os.path.join(result_dir,
                                                                 f"x0_{batch * batch_size + i}_{sample_config.y}_binary.jpg")
                            cv2.imwrite(y0_binary_path, np.array(y0_binary[i].cpu()).squeeze())
                        elif sample_config.y_ternary is True:
                            y0_ternary_path = os.path.join(result_dir,
                                                                  f"x0_{batch * batch_size + i}_{sample_config.y}_ternary.jpg")
                            cv2.imwrite(y0_ternary_path, np.array(y0_ternary[i].cpu()).squeeze())


                elif sample_config.result_type == 'sequence':
                    # 保存去噪过程
                    save_t_list = sorted(
                        set(np.linspace(1, len(xts) - 1, sample_config.save_num, dtype=int)).union({0}))
                    for i in tqdm(save_t_list, desc='saving_xts'):
                        # 保存xts
                        # xts[i][j]保存第j个样本去噪i步后得到的结果
                        # 遍历时间步，等间隔保存multi_sample_config.save_num个时间步的结果
                        xt = xts[i]
                        yt = yts[i]
                        xt = dbtgen_inverse_transform(img_size=sample_config.img_size)(xt)
                        yt = dbtgen_inverse_transform(img_size=sample_config.img_size)(yt)
                        for j in range(xts[i].shape[0]):
                            # 遍历样本
                            xts_result_dir = os.path.join(result_dir, f"xts_{j + batch * batch_size}")
                            if not os.path.exists(xts_result_dir):
                                os.makedirs(xts_result_dir)
                            x_path = os.path.join(xts_result_dir,
                                                  f"xt_{j + batch * batch_size}_{i - 1}_{sample_config.x}.jpg")
                            cv2.imwrite(x_path, np.array(xt[j].cpu()).squeeze())
                            y_path = os.path.join(xts_result_dir,
                                                       f"xt_{j + batch * batch_size}_{i - 1}_{sample_config.y}.jpg")
                            cv2.imwrite(y_path, np.array(yt[j].cpu()).squeeze())
                    save_t_list = sorted(
                        set(np.linspace(0, len(x0s) - 1, sample_config.save_num, dtype=int)))
                    for i in tqdm(save_t_list, desc='saving_x0s'):
                        # 保存x0s
                        # x0s[i][j]保存第j个样本进行第(i+1)步去噪时预测得到的x0
                        # 遍历时间步，等间隔保存multi_sample_config.save_num个时间步的结果
                        x0 = x0s[i]
                        y0 = y0s[i]
                        if sample_config.x_binary is True:
                            thre_weights = sample_config.binary_thre_weights
                            x0_binary = mask_to_binary(x0.to(device), thre_weights).to(dtype=torch.float)
                            x0_binary = dbtgen_inverse_transform(img_size=sample_config.img_size)(
                                x0_binary)
                        elif sample_config.x_ternary is True:
                            thre_weights = sample_config.ternary_thre_weights
                            x0_ternary = mask_to_ternary(x0.to(device), thre_weights).to(dtype=torch.float)
                            x0_ternary = dbtgen_inverse_transform(img_size=sample_config.img_size)(
                                x0_ternary)
                        if sample_config.y_binary is True:
                            thre_weights = sample_config.binary_thre_weights
                            y0_binary = mask_to_binary(y0.to(device), thre_weights).to(dtype=torch.float)
                            y0_binary = dbtgen_inverse_transform(img_size=sample_config.img_size)(
                                y0_binary)
                        elif sample_config.y_ternary is True:
                            thre_weights = sample_config.ternary_thre_weights
                            y0_ternary = mask_to_ternary(y0.to(device), thre_weights).to(
                                dtype=torch.float)
                            y0_ternary = dbtgen_inverse_transform(img_size=sample_config.img_size)(
                                y0_ternary)
                        x0 = dbtgen_inverse_transform(img_size=sample_config.img_size)(x0)
                        y0 = dbtgen_inverse_transform(img_size=sample_config.img_size)(y0)
                        for j in range(x0s[i].shape[0]):
                            x0s_result_dir = os.path.join(result_dir, f"x0s_{j + batch * batch_size}")
                            if not os.path.exists(x0s_result_dir):
                                os.makedirs(x0s_result_dir)
                            x_path = os.path.join(x0s_result_dir,
                                                  f"x0_{j + batch * batch_size}_{i}_{sample_config.x}.jpg")
                            y_path = os.path.join(x0s_result_dir,
                                                       f"x0_{j + batch * batch_size}_{i}_{sample_config.y}.jpg")
                            cv2.imwrite(x_path, np.array(x0[j].cpu()).squeeze())
                            cv2.imwrite(y_path, np.array(y0[j].cpu()).squeeze())
                            if sample_config.x_binary is True:
                                x0_binary_path = os.path.join(x0s_result_dir,
                                                              f"x0_{j + batch * batch_size}_{i}_{sample_config.x}_binary.jpg")
                                cv2.imwrite(x0_binary_path, np.array(x0_binary[j].cpu()).squeeze())
                            elif sample_config.x_ternary is True:
                                x0_ternary_path = os.path.join(x0s_result_dir,
                                                               f"x0_{j + batch * batch_size}_{i}_{sample_config.x}_ternary.jpg")
                                cv2.imwrite(x0_ternary_path, np.array(x0_ternary[j].cpu()).squeeze())
                            if sample_config.y_binary is True:
                                y0_binary_path = os.path.join(x0s_result_dir,
                                                                     f"x0_{j + batch * batch_size}_{i}_{sample_config.y}_binary.jpg")
                                cv2.imwrite(y0_binary_path, np.array(y0_binary[j].cpu()).squeeze())
                            elif sample_config.y_ternary is True:
                                y0_ternary_path = os.path.join(x0s_result_dir,
                                                                      f"x0_{j + batch * batch_size}_{i}_{sample_config.y}_ternary.jpg")
                                cv2.imwrite(y0_ternary_path, np.array(y0_ternary[j].cpu()).squeeze())
