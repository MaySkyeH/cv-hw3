import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import ToPILImage


# 定义SimCLR数据增强变换
class SimCLRTransform:
    def __init__(self, size=32):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size),  # 随机裁剪并调整大小
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # 随机改变亮度、对比度、饱和度和色调
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),  # 20%的概率将图像转换为灰度图
            transforms.GaussianBlur(kernel_size=3),  # 应用高斯模糊
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 标准化
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)  # 对同一图像应用两次变换
    
# 定义SimCLR数据集类
class SimCLRCIFAR10(Dataset):
    def __init__(self, transform=None):
        self.dataset = train_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)  # 返回数据集长度

    def __getitem__(self, idx):
        x, _ = self.dataset[idx]
        pil_image = ToPILImage()(x)
        return self.transform(pil_image)  # 返回两次变换后的图像
    
    # 定义SimCLR模型
class SimCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        super(SimCLR, self).__init__()
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])  # 使用基础模型的编码器部分（去掉最后一层）
        self.projection_head = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 2048),  # 投影头的第一层，全连接层
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(2048, out_dim)  # 投影头的第二层，全连接层
        )

    def forward(self, x):
        h = self.encoder(x)  # 获取编码器输出
        h = h.squeeze()  # 去除多余的维度
        z = self.projection_head(h)  # 通过投影头
        return F.normalize(z, dim=1)  # 返回归一化的向量
    
# 定义NT-Xent损失函数
class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)  # 余弦相似度函数

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)  # 对角线为0
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)  # 拼接正样本对

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature  # 计算相似度并除以温度参数

        sim_i_j = torch.diag(sim, self.batch_size)  # 正样本对相似度
        sim_j_i = torch.diag(sim, -self.batch_size)
        positives = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # positives = torch.cat([sim[i, self.batch_size + i].unsqueeze(0) for i in range(self.batch_size)] + 
        #                       [sim[self.batch_size + i, i].unsqueeze(0) for i in range(self.batch_size)], dim=0)

        negatives = sim[self.mask].reshape(N, -1)  # 负样本对相似度

        labels = torch.zeros(N).to(self.device).long()  # 标签，全为0

        logits = torch.cat((positives, negatives), dim=1)  # 拼接正负样本对
        loss = self.criterion(logits, labels)  # 计算损失
        loss /= N  # 平均损失

        return loss
    
# 定义线性分类器
class LinearClassifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super(LinearClassifier, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(512, num_classes)  # ResNet-18编码器的输出维度为512

    def forward(self, x):
        with torch.no_grad():
            h = self.encoder(x)  # 冻结编码器部分
        h = torch.flatten(h, start_dim=1)
        return self.fc(h)  # 通过线性分类器
   