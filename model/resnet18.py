import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

from config import config

# 加载预训练的 ResNet18 模型
# model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
# model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# 获取预训练模型的第一层卷积权重
pretrained_conv1_weight = model.conv1.weight.clone()

# 修改第一层卷积层的输入通道数为 18 通道
# 新的卷积层，输入通道为 18，输出通道仍为 64
model.conv1 = nn.Conv2d(len(config['layers']) * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
# model.conv1.in_channels = len(config['layers']) * 3

# 将预训练权重复制到新的卷积层的前3个通道
with torch.no_grad():
    model.conv1.weight[:, :3, :, :] = pretrained_conv1_weight  # 复制预训练的前3通道的权重
    if model.conv1.weight.shape[1] > 3:
        # 对剩余的 15 个通道使用相同的权重，或进行其他初始化方式
        model.conv1.weight[:, 3:, :, :] = pretrained_conv1_weight.mean(dim=1, keepdim=True)
# nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')


# 修改最后一个全连接层的输出类别为 2（原来为 1000 类）
model.fc = nn.Linear(model.fc.in_features, 2)

if __name__ == '__main__':
    print(model)

