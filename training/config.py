import os


class Config:
    def __init__(self):
        # 数据路径
        self.train_data_path = r'C:\Users\wenxi\Documents\study\project\CT_noise\new\data\processed'  # 处理后的干净图像路径
        self.noisy_data_path = r'C:\Users\wenxi\Documents\study\project\CT_noise\new\data\noisy'  # 加噪后的图像路径

        # 训练超参数
        self.batch_size = 16
        self.learning_rate = 1e-3
        self.epochs = 50
        self.clip_value = 1.0  # 梯度裁剪的阈值
        self.save_interval = 5  # 每5轮保存一次模型

        # 图像尺寸 (补充的部分)
        self.img_size = (256, 256)  # 定义图像的宽高，确保与数据预处理保持一致

        # 模型保存路径
        self.model_save_path = r'C:\Users\wenxi\Documents\study\project\CT_noise\new\models\saved_models'

        # 确保保存路径存在
        os.makedirs(self.model_save_path, exist_ok=True)
