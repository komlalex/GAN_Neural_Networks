import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader 

from torchvision.datasets import ImageFolder 
from torchvision.utils import make_grid
import torchvision.transforms as T 

import matplotlib.pyplot as plt

from pathlib import Path



TRAIN_DIR = Path("data")
IMAGE_SIZE = 64 
BATCH_SIZE = 128 
STATS = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

train_ds = ImageFolder(TRAIN_DIR, transform=T.Compose([
    T.Resize(IMAGE_SIZE),
    T.CenterCrop(IMAGE_SIZE),
    T.ToTensor(), 
    T.Normalize(*STATS)
])) 

train_dl = DataLoader(train_ds, 
                    shuffle=True,
                    batch_size=BATCH_SIZE, 
                    pin_memory=True) 

def denorm(img_tensors):
    return img_tensors * STATS[1][0] + STATS[0][0]  

def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([]) 
    ax.imshow(make_grid( denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))

def show_batch(dl, nmax=64): 
    for images, _ in dl:
        show_images(images, nmax)
        break 

show_batch(train_dl)
plt.show()



