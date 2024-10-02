import torch
import torch.nn as nn
from torchvision.models import convnext_tiny
from torchvision.models import ConvNeXt_Tiny_Weights

# 加载预训练的 ConvNeXt Tiny 模型
model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

# 获取预训练模型的第一层卷积权重
pretrained_conv1_weight = model.features[0][0].weight.clone()

# 修改第一层卷积层的输入通道数为 18 通道
# 新的卷积层，输入通道为 18，输出通道仍为 96
model.features[0][0] = nn.Conv2d(18, 96, kernel_size=4, stride=4, padding=0, bias=False)

# 将预训练权重复制到新的卷积层的前3个通道
with torch.no_grad():
    model.features[0][0].weight[:, :3, :, :] = pretrained_conv1_weight  # 复制预训练的前3通道的权重
    if model.features[0][0].weight.shape[1] > 3:
        # 对剩余的 15 个通道使用相同的权重，或进行其他初始化方式
        model.features[0][0].weight[:, 3:, :, :] = pretrained_conv1_weight.mean(dim=1, keepdim=True)

model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
