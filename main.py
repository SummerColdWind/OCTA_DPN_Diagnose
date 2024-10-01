from loader.loader import get_loader
from model.resnet18 import model
from tools.train import train


config = {
    'epoch_num': 1000,
    'lr': 2e-6,
    'batch_size': 8,
}

data_loader = get_loader(batch_size=config['batch_size'])
# train(model, data_loader, config, log_name='lr_1e_5_batch_size_8')
train(model, data_loader, config)

