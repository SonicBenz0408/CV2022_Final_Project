import os 
import torch.nn as nn
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from dataset import Img_Dataset
from pyramid import PyramidNet
from loss import WingLoss, NMELoss
import argparse
from PIL import Image

parser = argparse.ArgumentParser(description='Facial Landmark Detection')

parser.add_argument('--model_path', type=str, default=None, help='path of model checkpoint')

args = parser.parse_args()


if args.model_path == None:
    raise ValueError("Should include model path.")

val_path = "./data/aflw_val"
test_path = "./data/aflw_test"

output_path = "./output"
sol_path = os.path.join(output_path, "solution.txt")

# clean
with open(sol_path, "w"):
    pass

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

val_tfms = transforms.Compose([
    #transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_set = Img_Dataset(val_path, val_tfms)
test_set = Img_Dataset(test_path, val_tfms, have_anno=False)
val_names = val_set.names
test_names = test_set.names

print("Dataset complete!")

# Data loader
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

print("Dataloader complete!")

# Preparation
model = PyramidNet()
model.load_state_dict(torch.load(args.model_path))
evaluation = NMELoss() 

model.eval()
NME_loss = 0.

plt.figure()
# Validation
print("Validation Start!")

index = 0
for image, coords in tqdm(val_loader):
    
    with torch.no_grad():
        output = model(image)

        NME = evaluation(output, coords)
        image = image[0].numpy()
        gt = coords[0].numpy()
        predict = output[0].numpy()

        plt.clf()
        plt.imshow(Image.open(val_set.imgs[index]))
        plt.scatter(gt[:, 0], gt[:, 1], c="g", s=2)
        plt.scatter(predict[:, 0], predict[:, 1], c="r", s=2)
        plt.savefig(os.path.join(output_path, "val", val_names[index]))

        NME_loss += NME.item()
    
    index += 1

NME_loss /= len(val_set)

print(f'NME: {NME_loss:.7f}')

# Testing
print("Testing Start!")

index = 0
with open(sol_path, "w") as sol:
    for image, _ in tqdm(test_loader):
        
        with torch.no_grad():
            output = model(image)

            image = image[0].numpy()
            predict = output[0].numpy()

            w_string = test_names[index]
            for x, y in predict:
                w_string += f' {x:.4f} {y:.4f}'

            w_string += "\n"
            sol.write(w_string)
            
            plt.clf()
            plt.imshow(Image.open(test_set.imgs[index]))
            plt.scatter(predict[:, 0], predict[:, 1], c="r", s=2)
            plt.savefig(os.path.join(output_path, "test", test_names[index]))
        
        index += 1