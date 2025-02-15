import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.unet import UNet
from models.utils import calculate_psnr, clip_gradients
from training.data_loader import XRayDataset
from training.config import Config



def train():
    # 加载配置
    config = Config()

    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建数据集
    train_dataset = XRayDataset(config.train_data_path, config.noisy_data_path, config.img_size)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    # 初始化模型
    model = UNet(in_channels=1, out_channels=1).to(device)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # 训练过程
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        total_psnr = 0.0

        # 添加进度条
        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.epochs}'):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            clip_gradients(model, config.clip_value)
            optimizer.step()

            psnr = calculate_psnr(outputs, targets)
            total_psnr += psnr.item()
            running_loss += loss.item()

        print(
            f"Epoch [{epoch + 1}/{config.epochs}], Loss: {running_loss / len(train_loader):.4f}, PSNR: {total_psnr / len(train_loader):.2f}")

        # 保存模型
        if (epoch + 1) % config.save_interval == 0:
            torch.save(model.state_dict(), f"{config.model_save_path}/unet_epoch_{epoch + 1}.pth")

    print("Training completed.")


if __name__ == "__main__":
    train()
