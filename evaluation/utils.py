import torch
import torch.nn.functional as F

# 计算 PSNR（峰值信噪比）
def calculate_psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return torch.tensor(100.0)  # 避免除零错误
    return 20 * torch.log10(max_val / torch.sqrt(mse))

# 计算 SSIM（结构相似性指数）
def calculate_ssim(pred, target, window_size=11, C1=0.01**2, C2=0.03**2):
    """
    计算 SSIM，使用一个滑动窗口的方法。
    这里使用均值、方差和协方差的计算方式来模拟SSIM公式。
    """
    pred = pred.squeeze(0).squeeze(0)  # 去掉 batch 维度和通道维度
    target = target.squeeze(0).squeeze(0)

    mu_x = torch.mean(pred)
    mu_y = torch.mean(target)
    sigma_x = torch.var(pred)
    sigma_y = torch.var(target)
    sigma_xy = torch.mean((pred - mu_x) * (target - mu_y))

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)

    return numerator / denominator
