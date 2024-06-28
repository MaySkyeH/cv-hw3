# cv-hw3
# 数据下载
1. CIFAR-10数据集
```
import torch
from torchvision import datasets, transforms
transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
train_dataset = datasets.CIFAR10(root='./cv-hw3/train', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./cv-hw3/test', train=False, download=True, transform=transform)
torch.save(test_dataset, 'cv-hw3/test/test_dataset.pth')
```

2. CIFAR-100数据集
```
import torch
from torchvision import datasets, transforms
transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
CIFAR100_dataset = datasets.CIFAR100(root='./cv-hw3/test', train=False, download=True, transform=transform)
torch.save(CIFAR100_dataset, 'cv-hw3/test/CIFAR100_dataset.pth')
```

3. ImageNet数据集
https://image-net.org/challenges/LSVRC/2012/index.php#

# 模型训练
