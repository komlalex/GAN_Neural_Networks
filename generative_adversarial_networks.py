import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader 

from torchvision.datasets import ImageFolder 
import torchvision.transforms as T 

from pathlib import Path



TRAIN_DIR = Path("data")
print(type(TRAIN_DIR))
IMAGE_SIZE = 64 
BATCH_SIZE = 128 
STATS = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

train_ds = ImageFolder(TRAIN_DIR, transform=T.Compose([
    T.Resize(IMAGE_SIZE),
    T.CenterCrop(IMAGE_SIZE),
    T.ToTensor(), 
    T.Normalize(*STATS)
])) 

print(train_ds[0])


