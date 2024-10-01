import torch
import torchvision


model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, 1)
# model.add_module('prob', torch.nn.Sigmoid())


if __name__ == '__main__':
    print(model)

