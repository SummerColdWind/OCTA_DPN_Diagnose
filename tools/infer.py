from config import config
from loader.loader import get_loader, transform_val
from utils.common import load_image

import os
import torch
import colorama
import numpy as np
from sklearn.metrics import roc_auc_score


colorama.init(autoreset=True)

layers = config['layers']
root = config['root']
device = torch.device(config['device'])

def infer(model):
    _, val_loader = get_loader()
    model.to(device)
    model.eval()

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            positive_probs = probs[:, 1]
            all_probs.extend(positive_probs.cpu().numpy())  # 收集所有预测概率
            all_labels.extend(labels.cpu().numpy())  # 收集所有真实标签

    # 计算AUC
    roc_auc = roc_auc_score(all_labels, all_probs)
    print(colorama.Fore.YELLOW + f'AUC: {roc_auc:.4f}', end='\t')

    return all_probs, all_labels


def create_inputs(sample):
    """ 创建符合格式的输入 """
    dir_ = os.path.join(root, 'OCTA')
    images = os.listdir(os.path.join(dir_, sample))
    images = [load_image(os.path.join(dir_, sample, file)) for file in images if file in layers]
    images = [transform_val(i) for i in images]
    image = torch.cat(images, dim=0)
    image = image.unsqueeze(0)
    image = image.to(device)

    return image

def infer_single(model, sample):
    """ 推理单个样本 """
    model.to(device)
    model.eval()

    image = create_inputs(sample)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        positive_probs = probs[:, 1]

    return positive_probs[0].item()



