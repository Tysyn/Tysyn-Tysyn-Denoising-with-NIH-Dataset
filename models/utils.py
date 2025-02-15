import torch
import torch.nn as nn

# 处理卷积块，便于在U-Net中复用
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

# 计算PSNR（峰值信噪比）
def calculate_psnr(pred, target, max_val=255):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:  # Avoid division by zero
        return 100
    return 20 * torch.log10(max_val / torch.sqrt(mse))

# 计算SSIM（结构相似性指数）
def calculate_ssim(pred, target):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = torch.mean(pred)
    mu_y = torch.mean(target)
    sigma_x = torch.var(pred)
    sigma_y = torch.var(target)
    sigma_xy = torch.mean((pred - mu_x) * (target - mu_y))
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return numerator / denominator

# 上采样的卷积层
def upconv(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

# 处理跳跃连接
def crop_concat(x, skip):
    """裁剪skip连接，使其与x的形状一致，方便拼接"""
    return torch.cat([x, skip], dim=1)

# 处理反向传播时的梯度裁剪
def clip_gradients(model, clip_value=1.0):
    """在训练过程中裁剪梯度，防止梯度爆炸"""
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data = torch.clamp(param.grad.data, -clip_value, clip_value)
