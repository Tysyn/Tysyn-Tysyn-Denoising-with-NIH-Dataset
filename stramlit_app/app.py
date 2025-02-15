import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2
from models.unet import UNet
from evaluation.utils import calculate_psnr, calculate_ssim
from training.config import Config
import os

# 读取配置
config = Config()

# 加载训练好的模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=1, out_channels=1).to(device)
model.load_state_dict(torch.load(config.model_eval_path, map_location=device))
model.eval()

# 设置页面标题
st.title("医学图像去噪模型")

# 文件上传组件
uploaded_file = st.file_uploader("上传医学X光图像", type=["png", "jpg", "jpeg"])

# 判断是否上传文件
if uploaded_file is not None:
    # 显示上传的图像
    image = Image.open(uploaded_file)
    st.image(image, caption="上传的图像", use_column_width=True)

    # 转换图像为 numpy 数组并调整尺寸
    img = np.array(image.convert('L'))  # 转为灰度图像
    img = cv2.resize(img, (config.img_size))  # 调整为模型输入大小

    # 归一化图像
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    img = img.to(device)

    # 按钮触发去噪
    if st.button("去噪"):
        with torch.no_grad():
            denoised_img = model(img)

        # 转换去噪后的图像为 numpy 数组
        denoised_img = denoised_img.squeeze().cpu().numpy()

        # 显示去噪后的图像
        st.image(denoised_img, caption="去噪后的图像", use_column_width=True)

        # 计算 PSNR 和 SSIM
        psnr = calculate_psnr(denoised_img, img.squeeze().cpu().numpy())
        ssim = calculate_ssim(denoised_img, img.squeeze().cpu().numpy())

        st.write(f"PSNR: {psnr:.2f} dB")
        st.write(f"SSIM: {ssim:.4f}")

