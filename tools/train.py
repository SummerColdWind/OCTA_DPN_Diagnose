from torch.utils.tensorboard import SummaryWriter

import os
import torch
import torchvision
import colorama
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

colorama.init(autoreset=True)


default_config = {
    'epoch_num': 100,
    'lr': 1e-4,
    'batch_size': 8,
}




def train(model, data_loader, config=None, log_name=None):
    if config is None:
        config = {}
    config = {**default_config, **config}
    epoch_num, lr, batch_size = config['epoch_num'], config['lr'], config['batch_size']

    if log_name is None:
        log_name = str(datetime.now())[-6:]
    writer = SummaryWriter(f'logs/{log_name}')
    print(colorama.Fore.BLACK + colorama.Back.RED + colorama.Style.BRIGHT + 'tensorboard --logdir=logs')

    train_loader, val_loader, weights = data_loader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(colorama.Fore.RED + f'Device: {device}')
    model = model.to(device)
    loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
    # loss_func = torch.nn.CrossEntropyLoss(weight=weights)
    loss_func.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epoch_num):
        print(colorama.Fore.GREEN + f'--- Epoch {epoch} ---')
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            labels = labels
            opt.zero_grad()

            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(colorama.Fore.BLUE + f'Loss: {total_loss / len(train_loader)}')
        writer.add_scalar('Loss', total_loss, epoch)

        model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
                outputs = model(images)
                # print(outputs)
                # probs = torch.sigmoid(outputs).cpu().numpy()  # 将概率转换为 numpy 数组
                # print(probs)
                all_probs.extend(outputs.cpu().numpy())  # 收集所有预测概率
                all_labels.extend(labels.cpu().numpy())  # 收集所有真实标签

        # 计算 ROC-AUC 分数
        roc_auc = roc_auc_score(all_labels, all_probs)
        print(colorama.Fore.YELLOW + f'ROC-AUC: {roc_auc:.4f}')
        writer.add_scalar('AUC', roc_auc, epoch)

    if not os.path.exists(f'output/{log_name}'):
        os.makedirs(f'output/{log_name}')

    torch.save(model.state_dict(), f'output/{log_name}/last.pth')
    writer.close()
