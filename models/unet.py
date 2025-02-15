import torch
import torch.nn as nn


# U-Net模型定义
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()

        # 编码器（下采样）
        self.encoder = nn.ModuleList()
        for feature in features:
            self.encoder.append(self.double_conv(in_channels, feature))
            in_channels = feature

        # 中间层（bottleneck）
        self.bottleneck = self.double_conv(features[-1], features[-1] * 2)

        # 解码器（上采样）
        self.decoder = nn.ModuleList()
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(self.double_conv(feature * 2, feature))

        # 最后一层输出
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    # 双卷积层模块
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skip_connections = []

        # 编码路径
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        # 中间层
        x = self.bottleneck(x)

        # 解码路径
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)  # 上采样
            skip_connection = skip_connections[idx // 2]

            # 处理不同大小的跳跃连接
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](concat_skip)

        return self.final_conv(x)


# 模型测试
if __name__ == "__main__":
    model = UNet(in_channels=1, out_channels=1)
    x = torch.randn((1, 1, 256, 256))  # 模拟输入
    preds = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {preds.shape}")
