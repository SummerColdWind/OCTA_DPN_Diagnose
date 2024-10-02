from model.resnet18 import model
from tools.train import train

import torch
import random
import numpy as np

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)


train(model)


