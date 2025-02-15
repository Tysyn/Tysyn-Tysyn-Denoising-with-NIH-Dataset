# CT 图像去噪 - 基于 U-Net 和 NIH Chest X-ray Dataset

## 简介

本项目实现了基于 U-Net 网络的 CT 图像去噪功能。使用 NIH Chest X-ray Dataset 的简化版作为训练数据，旨在演示如何使用深度学习方法去除医学图像中的噪声，提升图像质量。

## 课程内容

本项目是[医学人工智能课程]的一部分，主要涉及以下内容：

*   U-Net 网络结构原理与实现
*   医学图像去噪的基本概念与方法
*   NIH Chest X-ray Dataset 数据集的加载与预处理
*   使用 PyTorch 搭建 U-Net 模型
*   模型训练、验证与测试
*   去噪效果评估与结果分析
*   使用 Streamlit 部署模型

## 数据集

*   **名称：** NIH Chest X-ray Dataset (Sample)
*   **来源：** [Kaggle - NIH Chest X-rays](https://www.kaggle.com/datasets/nih-chest-xrays/sample?resource=download)
*   **描述：** 本项目使用 NIH Chest X-ray Dataset 的简化版本，包含 [5606] 张胸部 X 光图像。由于完整数据集较大，本项目使用简化版进行演示。

## 数据预处理

本项目使用 `data_processing.py` 脚本对数据进行预处理。具体步骤如下：

1.  **加载原始图像：** 从 `data/raw` 文件夹加载原始图像。图像被转换为灰度图，并调整大小为 256x256 像素。

2.  **添加高斯噪声：** 对加载的图像添加高斯噪声，均值为 0，标准差为 25。噪声图像用于模拟真实的 CT 图像噪声。

3.  **保存噪声图像：** 将添加噪声后的图像保存到 `data/noisy` 文件夹。

4.  **保存原始图像：** 将调整大小后的原始图像保存到 `data/processed` 文件夹。这些图像将作为训练 U-Net 模型的ground truth。

## 模型

*   **结构：** U-Net (定义于 `unet.py`)
    *   **编码器：** 由多个双卷积层 (double_conv) 和最大池化层 (MaxPool2d) 组成，用于提取图像特征。双卷积层包含两个卷积层、批量归一化层 (BatchNorm2d) 和 ReLU 激活函数。
    *   **解码器：** 由多个转置卷积层 (ConvTranspose2d) 和双卷积层 (double_conv) 组成，用于恢复图像分辨率。
    *   **跳跃连接：** 将编码器和解码器中对应层的特征图连接起来，以保留更多细节信息。
    *   **中间层 (Bottleneck):** 连接编码器和解码器，使用双卷积层。
*   **实现：** PyTorch
*   **详细结构:**
    
    ```
    class UNet(nn.Module):
        def __init__(self, in_channels=1, out_channels=1, features=[128][256][512]):
            super(UNet, self).__init__()
            # ... (模型定义)
    ```

## 训练

*   **脚本：** `train.py`
*   **数据集：** 使用 `data_loader.py` 中的 `XRayDataset` 类加载数据。该类从 `data/processed` 和 `data/noisy` 文件夹读取干净和噪声图像，并将其转换为 PyTorch Tensor。
*   **配置：** 训练参数定义在 `config.py` 中，包括：
    
    *   `batch_size`: 16
    *   `learning_rate`: 1e-3
    *   `epochs`: 50
    *   `clip_value`: 1.0 (梯度裁剪阈值)
    *   `img_size`: (256, 256)
    *   `model_save_path`: 模型保存路径
*   **损失函数：** Mean Squared Error (MSE)
*   **优化器：** Adam
*   **训练过程：**
    
    *   每个 epoch 遍历训练数据集，计算损失并更新模型参数。
    *   使用梯度裁剪防止梯度爆炸。
    *   每 `save_interval` 个 epoch 保存模型到 `model_save_path`。
*   **关键代码:**
    ```
    # 训练过程 (train.py)
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        total_psnr = 0.0
    
        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.epochs}'):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            clip_gradients(model, config.clip_value)
            optimizer.step()
            # ...
    ```

## 评估

*   **脚本：** `evaluation.py`
*   **指标：**
    *   峰值信噪比 (PSNR): 使用 `utils.py` 中的 `calculate_psnr` 函数计算。
    *   结构相似性指数 (SSIM): 使用 `utils.py` 中的 `calculate_ssim` 函数计算。
*   **过程：**
    *   加载训练好的模型。
    *   遍历测试数据集，使用模型生成去噪图像。
    *   计算去噪图像和原始干净图像的 PSNR 和 SSIM。
    *   计算平均 PSNR 和 SSIM。
*   **结果：**
    *   Average PSNR:  27.5 dB (在测试集上, 模型实现了平均 27.5 dB 的 PSNR。*请注意，由于训练时间限制，这些结果可能未达到最优。*）
    *   Average SSIM: 0.80(在测试集上, 模型实现了平均 0.80 的 SSIM。*请注意，由于训练时间限制，这些结果可能未达到最优。*)

## 在线演示

*   **脚本:** `app.py`
*   **描述:** 使用 Streamlit 框架创建一个简单的 Web 应用程序，允许用户上传图像并使用训练好的模型进行去噪。
*   **功能:**
    
    *   用户可以上传医学 X 光图像 (png, jpg, jpeg)。
    *   应用程序将上传的图像显示在网页上。
    *   用户可以点击“去噪”按钮触发去噪过程。
    *   应用程序将去噪后的图像显示在网页上。
    *   应用程序计算并显示去噪图像的 PSNR 和 SSIM 值。
*   **运行方式:**
    ```
    streamlit run app.py
    ```
*   **注意:** 确保已安装 Streamlit (`pip install streamlit`)。

## 代码结构

CT_Denoise/
├── data/ # 数据相关文件夹
│ ├── noisy/ # 添加噪声后的图像
│ ├── processed/ # 处理后的干净图像
│ ├── raw/ # 原始图像数据
│ │ ├── data_processing.py # 数据预处理脚本
│ │ └── sample_labels.csv # 示例标签文件（如有需要）
├── evaluation/ # 模型评估相关文件夹
│ ├── evaluate.py # 模型评估脚本
│ └── utils.py # 辅助函数（如 PSNR 和 SSIM 的计算）
├── logs/ # 日志文件夹（如训练或评估日志）
├── models/ # 模型相关文件夹
│ ├── saved_models/ # 保存的模型权重
│ ├── unet.py # U-Net 模型定义
│ └── utils.py # 模型相关的辅助函数
├── streamlit_app/ # Streamlit 应用相关文件夹
│ ├── app.py # Streamlit 应用主脚本，用于在线演示去噪效果
│ └── utils.py # Streamlit 应用的辅助函数
├── training/ # 训练相关文件夹
│ ├── models/ # 与训练相关的模型代码
│ │ ├── config.py # 配置文件，定义训练参数和路径
│ │ ├── data_loader.py # 数据加载器，处理数据集的读取和预处理
│ │ └── train.py # 模型训练脚本
├── README.md # 项目说明文档
└── requirements.txt # 项目依赖列表

## 如何运行

1.  **克隆代码仓库：**

    ```
    git clone https://github.com/Tysyn/Tysyn-Tysyn-Denoising-with-NIH-Dataset.git
    ```

2.  **安装依赖：**

    ```
    pip install -r requirements.txt
    ```

3.  **准备数据集：**

    *   从 [Kaggle - NIH Chest X-rays](https://www.kaggle.com/datasets/nih-chest-xrays/sample?resource=download) 下载 NIH Chest X-ray Dataset (Sample)
    *   将数据集解压到 `data/raw` 目录下

4.  **运行数据预处理脚本：**

    ```
    python data_processing.py
    ```

5.  **运行训练脚本：**

    ```
    python training/train.py
    ```

6.  **运行评估脚本：**

    ```
    python evaluation/evaluate.py
    ```
7.  **运行 Streamlit 应用：**
    
     ```
     streamlit run app.py
     ```
