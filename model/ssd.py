import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

# A pretrained model trained with ImageNet:
model = torchvision.models.resnet50(pretrained=True)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

composed = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean, std)])

