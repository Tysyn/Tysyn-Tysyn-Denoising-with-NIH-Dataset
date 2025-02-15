import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np


class XRayDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, img_size=(256, 256)):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.img_size = img_size
        self.clean_images = os.listdir(clean_dir)
        self.noisy_images = os.listdir(noisy_dir)

    def __len__(self):
        return len(self.clean_images)

    def __getitem__(self, idx):
        clean_image_path = os.path.join(self.clean_dir, self.clean_images[idx])
        noisy_image_path = os.path.join(self.noisy_dir, self.noisy_images[idx])

        # 读取图像
        clean_img = cv2.imread(clean_image_path, cv2.IMREAD_GRAYSCALE)
        noisy_img = cv2.imread(noisy_image_path, cv2.IMREAD_GRAYSCALE)

        # 调整图像大小
        clean_img = cv2.resize(clean_img, self.img_size)
        noisy_img = cv2.resize(noisy_img, self.img_size)

        # 转为Tensor
        clean_img = torch.tensor(clean_img, dtype=torch.float32).unsqueeze(0) / 255.0
        noisy_img = torch.tensor(noisy_img, dtype=torch.float32).unsqueeze(0) / 255.0

        return noisy_img, clean_img
