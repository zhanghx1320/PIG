import torch
from tqdm import tqdm
from PIG.utils.mask_to_discrete import mask_to_binary, mask_to_ternary


def compute_alpha(betas, t):
    assert betas.device == t.device, print(betas.device, t.device)
    betas = torch.cat([torch.zeros(1).to(betas.device), betas], dim=0)
    alphat = (1 - betas).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return alphat


def one_step_denoise(model, xt, cond, t, betas, t_next=None, eta=0.0):
    """
    单模型单步去噪，预测et_pred->计算x0_pred(->(t_next不为None时)计算xt_next)
    Args:
        model: 模型
        xt: t时刻的x
        t: 当前时刻
        t_next: 下一时刻，为None时不计算xt_next
        betas: betas
        cond: 条件
        eta: 方差参数

    Returns: x0_pred(, xt_next_pred)

    """

    assert xt.shape[0] == t.shape[0]
    model.eval()
    with torch.no_grad():
        device = xt.device
        t, betas, t_next, cond = t.to(device), betas.to(device), t_next.to(
            device) if t_next is not None else None, cond.to(device) if cond is not None else None
        et_pred = model(xt, cond, t)
        # 计算x0
        alpha_t = compute_alpha(betas, t.long())
        x0_pred = (xt - et_pred * (1 - alpha_t).sqrt()) / alpha_t.sqrt()

        if t_next is not None:
            # 计算xt_next
            alphat_next = compute_alpha(betas, t_next.long())
            c1 = eta * ((1 - alpha_t / alphat_next) * (1 - alphat_next) / (1 - alpha_t)).sqrt()
            c2 = ((1 - alphat_next) - c1 ** 2).sqrt()
            xt_next_pred = alphat_next.sqrt() * x0_pred + c1 * torch.randn_like(xt) + c2 * et_pred
    if t_next is not None:
        return x0_pred, xt_next_pred
    else:
        return x0_pred


def multi_denoise(x_model, y_model, xt, yt, t_list, betas, eta=0.0, x_binary=False,
                  x_ternary=False, binary_thre_weights=None, ternary_thre_weights=None):
    """
    多模型去噪，预测x_et_pred->预测x0_pred->预测y_et_pred->预测y0_pred
    Args:
        x_model: x模型
        y_model: y模型
        xt: t时刻的条件
        yt: t时刻的目标
        t_list: 时间列表
        betas: betas
        eta: 方差参数
        x_binary: 是否对条件进行二值化
        binary_thre_weights: 二值化阈值=thre_weights[0]*max+thre_weights[1]*min

    Returns: xts, yts, x0s, y0s

    """
    assert xt.shape == yt.shape
    assert xt.device == yt.device
    x_model.eval()
    y_model.eval()
    with torch.no_grad():
        device = xt.device
        betas = betas.to(device)
        img_num = xt.shape[0]
        t_next_list = [-1] + list(t_list[:-1])
        x0s = []  # 保存每个时间步预测的x_0，x0s[i][j]保存第j个样本进行第(i+1)步去噪时预测得到的x0
        xts = [xt]  # 保存每个时间步预测的x_prev，xts[i][j]保存第j个样本去噪i步后得到的x_prev，比x0s多一个(原始噪声)
        y0s = []
        yts = [yt]
        for i, j in tqdm(zip(reversed(t_list), reversed(t_next_list)), desc='multi_denoising', total=len(t_list)):
            t = (torch.ones(img_num) * i).to(device)
            t_next = (torch.ones(img_num) * j).to(device)
            xt = xts[-1].to(device)
            yt = yts[-1].to(device)
            x0, xt = one_step_denoise(x_model, xt, yt, t, betas, t_next=t_next, eta=eta)
            x0s.append(x0.to('cpu'))
            xts.append(xt.to('cpu'))

            if x_binary:
                x0 = mask_to_binary(x0, binary_thre_weights).to(torch.float)
            elif x_ternary:
                x0 = mask_to_ternary(x0, ternary_thre_weights).to(torch.float)
            y0, yt = one_step_denoise(y_model, yt, x0, t, betas, t_next=t_next, eta=eta)
            y0s.append(y0.to('cpu'))
            yts.append(yt.to('cpu'))
    return xts, yts, x0s, y0s
