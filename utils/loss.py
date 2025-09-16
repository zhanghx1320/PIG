import torch


def noise_estimation_loss(model, x0: torch.Tensor, cond, t: torch.LongTensor, e: torch.Tensor, betas: torch.Tensor,
                          keepdim=False, loss_weight_flag=False):
    alphas = (1 - betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    xt = x0 * alphas.sqrt() + e * (1.0 - alphas).sqrt()
    e_pred = model(xt, cond, t.float())
    if keepdim:
        # return (e - e_pred).square().sum(dim=(1, 2, 3))
        return (e - e_pred).square().mean(dim=(1, 2, 3)) if not loss_weight_flag else (
                (e - e_pred).square() * cond.sigmoid()).mean(dim=(1, 2, 3))
    else:
        # return (e - e_pred).square().sum(dim=(1, 2, 3)).mean(dim=0)
        return (e - e_pred).square().mean() if not loss_weight_flag else ((e - e_pred).square() * cond.sigmoid()).mean()
