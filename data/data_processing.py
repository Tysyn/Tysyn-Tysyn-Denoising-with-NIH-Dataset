import os
import shutil
import numpy as np
import cv2
from tqdm import tqdm

# 加载原始图像
def load_images_from_folder(folder, img_size=(256, 256)):
    images = []
    filenames = []
    for filename in tqdm(os.listdir(folder)):
        if filename.endswith('.png'):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, img_size)
                images.append(img)
                filenames.append(filename)
    return images, filenames

# 添加高斯噪声
def add_gaussian_noise(images, mean=0, std=25):
    noisy_images = []
    for img in images:
        noise = np.random.normal(mean, std, img.shape)
        noisy_img = img + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        noisy_images.append(noisy_img)
    return noisy_images

# 保存处理后的图像
def save_images(images, filenames, folder):
    for img, filename in zip(images, filenames):
        save_path = os.path.join(folder, filename)
        cv2.imwrite(save_path, img)

# 数据预处理流程
def preprocess_data(raw_folder=r'C:\Users\wenxi\Documents\study\project\CT_noise\new\data\raw', noisy_folder=r'C:\Users\wenxi\Documents\study\project\CT_noise\new\data\noisy', processed_folder=r'C:\Users\wenxi\Documents\study\project\CT_noise\new\data\processed'):
    print("Loading raw images...")
    images, filenames = load_images_from_folder(raw_folder)

    print("Adding Gaussian noise...")
    noisy_images = add_gaussian_noise(images)

    print("Saving noisy images...")
    save_images(noisy_images, filenames, noisy_folder)

    print("Saving processed (resized) clean images...")
    save_images(images, filenames, processed_folder)

    print("Data preprocessing completed.")

if __name__ == "__main__":
    preprocess_data()
