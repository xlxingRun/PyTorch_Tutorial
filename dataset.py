# 读取数据集，并进行归一化处理
import torch
import torchvision
import torchvision.transforms as transforms

# 定义数据处理的方式
# 将数据转化为tensor
"""
transforms.ToTensor() 
把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，
转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloatTensor
"""

"""
transforms.Normalize(mean, std)
给定均值(R, G, B) 方差(R, G, B)，将会把tensor正则化，Normalized_image=(image-mean)/std
"""
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 训练集
train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4,
                                           shuffle=True, num_workers=2)

# 测试集
test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4,
                                          shuffle=False, num_workers=2)

# 分类元组
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

