import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import cv2
import pandas as pd
import os

from utils.common import load_image

root = 'data/batch_1/clean'
layer = '视网膜血流'
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 归一化
])


class OCTADataset(Dataset):
    def __init__(self):
        self.labels = pd.read_csv(os.path.join(root, 'label.csv'), index_col=0)
        self.images = os.listdir(os.path.join(root, 'OCTA'))
        self.data = [
            (transform(load_image(os.path.join(root, 'OCTA', image, f'{layer}.jpg'))),
             torch.tensor(self.labels['label'][image], dtype=torch.long))
            for image in self.images
        ]
        counts = self.labels['label'].value_counts()
        # self.pos_weight = counts[1] / (counts[0] + counts[1])
        # self.neg_weight = counts[0] / (counts[0] + counts[1])
        self.pos_weight = counts[0] / counts[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def get_loader(batch_size=64):
    dataset = OCTADataset()
    length = len(dataset)
    threshold = int(length * .8)
    train_dataset, val_dataset = dataset[:threshold], dataset[threshold:]
    train_loader, val_loader = (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    )
    # return train_loader, val_loader, torch.tensor([dataset.neg_weight, dataset.pos_weight], dtype=torch.float32)
    return train_loader, val_loader, torch.tensor(dataset.pos_weight, dtype=torch.float32)
