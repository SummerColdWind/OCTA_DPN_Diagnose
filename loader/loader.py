import torch
import torchvision
import cv2
import pandas as pd
import os
import random

from torch.utils.data import DataLoader, Dataset
from config import config
from utils.common import load_image

root = config['root']
train_batch_size = config['train_batch_size']
val_batch_size = config['val_batch_size']

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 归一化
])

class OCTADataset(Dataset):
    def __init__(self):
        super().__init__()
        self.labels = pd.read_csv(os.path.join(root, 'label.csv'), index_col=0)
        self.data = []
        for sample in self.labels.index:
            dir_ = os.path.join(root, 'OCTA', sample)
            files = os.listdir(dir_)
            images = [transform(load_image(os.path.join(dir_, file))) for file in files]
            image = torch.cat(images, dim=0)  # shape: [18, H, W]
            label = torch.tensor(self.labels['label'][sample]).long()
            self.data.append((image, label))
        # random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def get_loader():
    dataset = OCTADataset()
    length = len(dataset)
    val_frac = config['val_frac']
    threshold = int(length * val_frac)
    train_dataset, val_dataset = dataset[:threshold], dataset[threshold:]
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


