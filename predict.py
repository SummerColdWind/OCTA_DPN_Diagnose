from model.resnet18 import model
from tools.infer import infer, infer_single
import torch


model.eval()
model.load_state_dict(torch.load('output/766111/best.pth'))
# all_probs, all_labels = infer(model)
# result = np.asarray([all_probs, all_labels]).T

result = infer_single(model, '冷传澔_右眼')
print(result)

