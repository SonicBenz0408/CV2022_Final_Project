import os 
import torch.nn as nn
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from dataset import Img_Dataset
from network import FacialLandmark, PyramidNet
from loss import WingLoss, NMELoss
import argparse


parser = argparse.ArgumentParser(description='Facial Landmark Detection')

parser.add_argument('--model_path', type=str, default=None, help='path of model checkpoint')

args = parser.parse_args()


if args.model_path == None:
    raise ValueError("Should include model path.")

val_path = "./data/aflw_val"

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

val_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_set = Img_Dataset(val_path, val_tfms)

print("Dataset complete!")

# Data loader
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

print("Dataloader complete!")

# Preparation
model = PyramidNet().cuda()
model.load_state_dict(torch.load(args.model_path))
evaluation = NMELoss() 

print("Validation Start!")

model.eval()
NME_loss = 0.
for image, coords in tqdm(val_loader):
    
    image, coords = image.cuda(), coords.cuda()

    with torch.no_grad():
        output = model(image)

        NME = evaluation(output, coords)
        NME_loss += NME.item()

NME_loss /= len(val_set)

print(f'NME: {NME_loss:.4f}')

image = image[0].cpu().numpy()
gt = coords[0].cpu().numpy()
predict = output[0].cpu().numpy()

plt.figure(0)
plt.imshow(np.transpose(image, (1, 2, 0)))
plt.scatter(gt[:, 0], gt[:, 1], c="r")

plt.figure(1)
plt.imshow(np.transpose(image, (1, 2, 0)))
plt.scatter(predict[:, 0], predict[:, 1], c="r")

plt.show()