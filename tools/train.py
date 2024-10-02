from torch.utils.tensorboard import SummaryWriter

import os
import torch
import torchvision
import colorama
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from config import config
from loader.loader import get_loader, get_class_weights

colorama.init(autoreset=True)



def train(model, log_name=None):
    epoch_num, lr = config['epoch_num'], config['lr']

    if log_name is None:
        log_name = str(datetime.now())[-6:]
    writer = SummaryWriter(f'logs/{log_name}')
    print(colorama.Fore.BLACK + colorama.Back.RED + colorama.Style.BRIGHT + 'tensorboard --logdir=logs')

    train_loader, val_loader = get_loader()

    device = torch.device(config['device'])
    print(colorama.Fore.RED + f'Device: {device}')

    model = model.to(device)
    # loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
    weights = get_class_weights()
    weights.to(device)
    loss_func = torch.nn.CrossEntropyLoss(weight=weights)
    loss_func.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epoch_num):
        print(colorama.Fore.GREEN + f'--- Epoch {epoch} ---', end='\t')
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()

            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(colorama.Fore.BLUE + f'Loss: {total_loss / len(train_loader):.5f}', end='\t')
        writer.add_scalar('Loss', total_loss, epoch)

        model.eval()
        all_probs = []
        all_labels = []
        acc_num = total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                preds = probs.argmax(dim=1)
                acc_num += torch.sum(preds == labels).cpu().item()
                total += len(labels)
                positive_probs = probs[:, 1]
                all_probs.extend(positive_probs.cpu().numpy())  # 收集所有预测概率
                all_labels.extend(labels.cpu().numpy())  # 收集所有真实标签

        # 计算准确率
        acc = acc_num / total * 100
        print(colorama.Fore.YELLOW + f'acc: {acc:.2f}%', end='\t')
        writer.add_scalar('acc', acc, epoch)

        # 计算AUC
        roc_auc = roc_auc_score(all_labels, all_probs)
        print(colorama.Fore.YELLOW + f'AUC: {roc_auc:.4f}', end='\t')
        writer.add_scalar('AUC', roc_auc, epoch)

        print()

    if not os.path.exists(f'output/{log_name}'):
        os.makedirs(f'output/{log_name}')

    torch.save(model.state_dict(), f'output/{log_name}/last.pth')
    writer.close()
