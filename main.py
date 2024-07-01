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
from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter
from simCLR import NTXentLoss, SimCLR, SimCLRCIFAR10, SimCLRTransform, LinearClassifier

print('训练自监督模型')
print('---------------------')
# 加载数据集
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
train_dataset = torch.load('train/train_dataset.pth')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)

# Hyperparameters
batch_size = 128
temperature = 0.5
epochs = 50
lr = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_transform = SimCLRTransform() 
Train_dataset = SimCLRCIFAR10(transform=train_transform)
Train_loader = DataLoader(Train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

# Model, Loss, Optimizer
base_model = models.resnet18(pretrained=False)
simclr_model = SimCLR(base_model, out_dim=128).to(device)

criterion = NTXentLoss(batch_size, temperature, device)
optimizer = optim.Adam(simclr_model.parameters(), lr=lr, weight_decay=1e-6)

writer = SummaryWriter('runs/simclr_loss')

# 训练循环
for epoch in range(epochs):  # 迭代指定的训练轮数
    simclr_model.train()  # 将模型设置为训练模式
    total_loss = 0  # 初始化总损失

    for i, (x_i, x_j) in enumerate(Train_loader):
        x_i, x_j = x_i.to(device), x_j.to(device)  # 将数据移动到GPU（如果可用）
        z_i, z_j = simclr_model(x_i), simclr_model(x_j)  # 将两个图像批量输入模型，获得它们的嵌入表示
        loss = criterion(z_i, z_j)  # 计算NT-Xent损失

        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数

        total_loss += loss.item()  # 累加当前批次的损失
        
        # 每10个批次记录一次loss到TensorBoard
        if i % 10 == 0:
            writer.add_scalar('training loss', loss.item(), epoch * len(train_loader) + i)

    avg_loss = total_loss / len(Train_loader)  # 计算平均损失
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")  # 打印当前轮次的平均损失

# 保存模型参数
torch.save(simclr_model.encoder.state_dict(), 'model/simclr_encoder.pth')

# 加载预训练的SimCLR模型的编码器部分的状态字典
encoder_state_dict = torch.load('model/simclr_encoder.pth')

# 加载预训练的编码器
base_model = models.resnet18(pretrained=False)
encoder = nn.Sequential(*list(base_model.children())[:-1])
encoder.load_state_dict(encoder_state_dict)

num_classes = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
linear_model = LinearClassifier(encoder, num_classes).to(device)

test_dataset = torch.load('test/test_dataset.pth')
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

# writer = SummaryWriter('runs0/CIFAR')

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(linear_model.fc.parameters(), lr=0.001, weight_decay=1e-4)

# 训练线性分类器
epochs = 60
for epoch in range(epochs):
    linear_model.train()
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = linear_model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # 每10个批次记录一次loss到TensorBoard
        if i % 10 == 0:
            writer.add_scalar('training loss', loss.item(), epoch * len(test_loader) + i)

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    # 评估模型在测试集上的accuracy
    linear_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = linear_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the linear classifier on the CIFAR-100 test images: {accuracy:.2f}%')

    # 记录accuracy到TensorBoard
    writer.add_scalar('accuracy', accuracy, epoch)