from dataset import Img_Dataset
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

train_path = "../data/synthetics_train"
val_path = "../data/aflw_val"

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tfms = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_set = Img_Dataset(train_path, train_tfms)
val_set = Img_Dataset(val_path, val_tfms)


# Hyperparameter
batch_size = 32

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

