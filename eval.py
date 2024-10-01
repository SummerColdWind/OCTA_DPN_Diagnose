from model.resnet18 import model
from loader.loader import transform
from utils.common import load_image

import torch
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from PIL import Image

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


model.load_state_dict(torch.load('output/lr_1e_5_batch_size_4/last.pth'))
model.eval()

cam_extractor = SmoothGradCAMpp(model)
image = Image.open(r'C:\Users\Qiao\Desktop\projects\OCTA\data\batch_1\clean\OCTA\刘延财_右眼\视网膜血流.jpg')
inputs = transform(image).unsqueeze(0)

output = model(inputs)
print(output)
pred_class = output.argmax(dim=1).item()
activation_map = cam_extractor(pred_class, output)
print(activation_map)
# 将 CAM 张量转换为 NumPy 格式
activation_map = activation_map[0].cpu().numpy()
activation_map = activation_map.squeeze(0)
# 归一化 CAM 值到 [0, 1] 范围
activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
print(activation_map)
print(activation_map.shape)
# 转换为 PIL 图像
activation_map = to_pil_image(activation_map)

# 使用 torchcam 提供的工具将 CAM 叠加到原始图像上
result = overlay_mask(image, activation_map, alpha=0.5)

# 显示叠加结果
plt.imshow(result)
plt.axis('off')
plt.show()
