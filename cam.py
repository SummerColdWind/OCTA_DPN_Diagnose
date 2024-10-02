from model.resnet18 import model
from tools.infer import create_inputs, infer_single
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask

import matplotlib.pyplot as plt
import torch


model.load_state_dict(torch.load('output/766111/best.pth'))
model.eval()


target_layers = [model.layer4[-1]]
sample = '于世华_右眼'
input_tensor = create_inputs(sample).cpu()

img = read_image(f'data/batch_1/clean/OCTA/{sample}/视网膜血流.jpg')
# input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


with SmoothGradCAMpp(model, input_shape=(6, 224, 224)) as cam_extractor:
    # Preprocess your data and feed it to the model
    out = model(input_tensor)
    print('预测类别为', out.squeeze(0).argmax().item())
    probs = torch.nn.functional.softmax(out, dim=1)
    positive_probs = probs[:, 1]
    print('阳性概率为', positive_probs[0].item())
    # Retrieve the CAM by passing the class index and the model output
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)


result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
plt.imshow(result)
plt.axis('off')
plt.tight_layout()
plt.show()
