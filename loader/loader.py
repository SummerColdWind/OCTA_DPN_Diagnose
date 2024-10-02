import torch
import torchvision
import pandas as pd
import os
import random

from torch.utils.data import DataLoader, Dataset
from config import config
from utils.common import load_image

root = config['root']
train_batch_size = config['train_batch_size']
val_batch_size = config['val_batch_size']

transform_val = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 归一化
])

transform_train = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((224, 224)),
    # torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 随机调整亮度和对比度
    # torchvision.transforms.RandomHorizontalFlip(),  # 随机水平翻转
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 归一化
])

layers = config['layers']




class OCTADataset(Dataset):
    def __init__(self, data, transform):
        super().__init__()
        self.transform = transform
        self.data = data
        # random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        images, label = self.data[item]
        images = [self.transform(i) for i in images]
        image = torch.cat(images, dim=0)
        return image, label

def load_data():
    labels = pd.read_csv(os.path.join(root, 'label.csv'), index_col=0)
    data = []
    for sample in labels.index:
        dir_ = os.path.join(root, 'OCTA', sample)
        try:
            files = os.listdir(dir_)
        except FileNotFoundError:
            continue
        images = [load_image(os.path.join(dir_, file)) for file in files if file in layers]
        label = torch.tensor(labels['label'][sample]).long()
        data.append((images, label))
    return data


def get_loader():
    data = load_data()
    length = len(data)
    val_frac = config['val_frac']
    threshold = int(length * (1 - val_frac))
    train_dataset, val_dataset = (
        OCTADataset(data[:threshold], transform_train),
        OCTADataset(data[threshold:], transform_val)
    )
    train_loader, val_loader = (
        DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    )
    return train_loader, val_loader

def get_class_weights():
    """ 分类权重，用于交叉熵损失函数等 """
    labels = pd.read_csv(os.path.join(root, 'label.csv'), index_col=0)
    counts = labels['label'].value_counts()
    return torch.tensor([1 / counts[0], 1 / counts[1]], dtype=torch.float)


