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
    epoch_num, lr, weight_decay = config['epoch_num'], config['lr'], config['weight_decay']

    if log_name is None:
        log_name = str(datetime.now())[-6:]
    writer = SummaryWriter(f'logs/{log_name}')
    print(colorama.Fore.BLACK + colorama.Back.RED + colorama.Style.BRIGHT + 'tensorboard --logdir=logs')

    train_loader, val_loader = get_loader()
    print(colorama.Fore.CYAN + f'Train size ~ {len(train_loader) * config["train_batch_size"]}')
    print(colorama.Fore.CYAN + f'Valid size ~ {len(val_loader) * config["val_batch_size"]}')

    device = torch.device(config['device'])
    print(colorama.Fore.RED + f'Device: {device}')

    model = model.to(device)
    # loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
    weights = get_class_weights()
    weights.to(device)
    loss_func = torch.nn.CrossEntropyLoss(weight=weights)
    loss_func.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epoch_num)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, 5, 0.8)

    best = 0.
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
        scheduler.step()
        print(colorama.Fore.BLUE + f'loss: {total_loss / len(train_loader):.5f}', end='\t')
        print(colorama.Fore.MAGENTA + f'lr: {scheduler.get_last_lr()[0]:.6f}', end='\t')
        writer.add_scalar('loss', total_loss, epoch)

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
        # print(colorama.Fore.YELLOW + f'acc: {acc:.2f}%', end='\t')
        writer.add_scalar('acc', acc, epoch)

        # 计算AUC
        roc_auc = roc_auc_score(all_labels, all_probs)
        print(colorama.Fore.YELLOW + f'AUC: {roc_auc:.4f}', end='\t')
        writer.add_scalar('AUC', roc_auc, epoch)

        if config['save'] and roc_auc > best:
            if not os.path.exists(f'output/{log_name}'):
                os.makedirs(f'output/{log_name}')
            torch.save(model.state_dict(), f'output/{log_name}/best.pth')

        best = max(roc_auc, best)
        print(colorama.Fore.RED + f'best AUC: {best:.4f}', end='\t')
        print()

    if config['save']:
        if not os.path.exists(f'output/{log_name}'):
            os.makedirs(f'output/{log_name}')

        torch.save(model.state_dict(), f'output/{log_name}/last.pth')
    writer.close()
