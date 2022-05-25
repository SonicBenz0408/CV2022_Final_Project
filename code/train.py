import os 
import torch.nn as nn
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from dataset import Img_Dataset
from network import FacialLandmark
from loss import WingLoss, NMELoss

train_path = "./data/synthetics_train"
val_path = "./data/aflw_val"

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

print("Dataset complete!")
# Hyperparameters
batch_size = 32
learning_rate = 0.001
max_epoch = 40

save_path = "./log"

os.makedirs(save_path, exist_ok=True)
# Data loader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

print("Dataloader complete!")

# Preparation
model = FacialLandmark().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = WingLoss()
evaluation = NMELoss() 

train_loss_curve, val_loss_curve = [], []
NME_curve = []

print("Training Start!")
for epoch in range(max_epoch):

    model.train()

    train_loss, val_loss = 0., 0.
    for image, coords in tqdm(train_loader):
        
        image, coords = image.cuda(), coords.cuda()

        output = model(image)

        loss = criterion(output, coords)
        
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    NME_loss = 0.
    for image, coords in tqdm(val_loader):
        
        image, coords = image.cuda(), coords.cuda()

        with torch.no_grad():
            output = model(image)
            loss = criterion(output, coords)
            val_loss += loss.item()

            NME = evaluation(output, coords)
            NME_loss += NME.item()

        torch.save(model.state_dict(), os.path.join(save_path, "last_model.pth"))
    
    train_loss /= len(train_set)
    val_loss /= len(val_set)
    NME_loss /= len(val_set)

    print(f'Epoch: {epoch+1}/{max_epoch}, Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}, NME: {NME_loss:.4f}')

    train_loss_curve.append(train_loss)
    val_loss_curve.append(val_loss)
    NME_curve.append(NME_loss)


x_axis = np.arange(max_epoch)
plt.figure(0)
plt.plot(x_axis, train_loss_curve, c="r")
plt.plot(x_axis, val_loss_curve, c="b")
plt.title("Training Curve")
plt.legend(["train loss", "val loss"])

plt.figure(1)
plt.plot(x_axis, NME_curve, c="r")
plt.title("NME Loss Curve")