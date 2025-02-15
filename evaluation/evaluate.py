import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import cv2
import numpy as np
from models.unet import UNet
from evaluation.utils import calculate_psnr, calculate_ssim
from training.data_loader import XRayDataset
from training.config import Config


def evaluate():
    # 加载配置
    config = Config()

    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载测试数据
    test_dataset = XRayDataset(config.test_data_path, config.noisy_test_data_path, config.img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 加载训练好的模型
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(config.model_eval_path, map_location=device))
    model.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0

    # 评估模型
    with torch.no_grad():
        for noisy_imgs, clean_imgs in test_loader:
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)

            # 生成去噪图像
            denoised_imgs = model(noisy_imgs)

            # 计算 PSNR 和 SSIM
            psnr = calculate_psnr(denoised_imgs, clean_imgs)
            ssim = calculate_ssim(denoised_imgs, clean_imgs)

            total_psnr += psnr.item()
            total_ssim += ssim.item()
            num_samples += 1

    # 计算平均 PSNR 和 SSIM
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples

    print(f"Model Evaluation Results:\nAverage PSNR: {avg_psnr:.2f} dB\nAverage SSIM: {avg_ssim:.4f}")


if __name__ == "__main__":
    evaluate()
