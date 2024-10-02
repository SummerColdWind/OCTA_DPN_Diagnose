config = {
    'device': 'cuda',  # 'cpu' or 'cuda'
    'root': 'data/batch_1/clean',
    'val_frac': 0.1,
    'shuffle': False,
    'train_batch_size': 64,
    'val_batch_size': 8,
    'epoch_num': 30,
    'lr': 1e-3,
    'weight_decay': 0,
    'layers': [
        '表层血流.jpg',
        # '脉络膜毛细血管.jpg',
        # '深层血流.jpg',
        '视网膜血流.jpg',
        # '玻璃体血流.jpg',
        # '无血管层血流.jpg',
    ],
    'save': True,
}

