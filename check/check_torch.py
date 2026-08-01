import torch
import torchvision

print("torch version:", torch.__version__)
print("torchvision version:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())