import os
import random
import torch.nn as nn
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from dataset import Img_Dataset
from pyramid import PyramidNet
from loss import WingLoss, NMELoss, WeightedL2Loss, CenterLoss

# fix random seeds for reproducibility
SEED = 7414
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

train_path = "./data/synthetics_train"
val_path = "./data/aflw_val"

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tfms = transforms.Compose([
    transforms.ColorJitter(brightness=0.1),#, saturation=0.2, hue=0.2),
    #transforms.GaussianBlur(5),
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
max_epoch = 30
noise_epoch = 1             
noise_size = 41
shifting = noise_size // 2

weights = [1.] * 27 + [20] * 9 + [1] * 24 + [20] * 8
weights = torch.FloatTensor(weights)
#print(weights)

white = torch.Tensor([[[2.2489]], [[2.4286]], [[2.6400]]]).repeat(1, noise_size, noise_size)
#white = torch.Tensor([2.2489, 2.4286, 2.6400]).repeat(384, 384, 1)
save_path = "./log/MobileNetv2_32centerloss"

os.makedirs(save_path, exist_ok=True)
# Data loader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

print("Dataloader complete!")

# Preparation
model = PyramidNet().cuda()
optimizer = torch.optim.Adam([{"params":model.parameters(), "initial_lr": learning_rate}], lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, len(train_loader), 0.88)

criterion = WingLoss()
centerLoss = CenterLoss()
#criterion = WeightedL2Loss(weights=weights)
#criterion = NMELoss()
evaluation = NMELoss()

train_loss_curve, val_loss_curve = [], []
NME_curve = []
print("Training Start!")
for epoch in range(max_epoch):

    model.train()

    train_loss, val_loss = 0., 0.
    if epoch < noise_epoch:
        for image, coords in tqdm(train_loader):
            
            image, coords = image.cuda(), coords.cuda()

            output = model(image)


            loss = criterion(output, coords) + centerLoss(output, coords)
            
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    else:
        for image, coords in tqdm(train_loader):
            
            # Patch mask
            
            random_idxs = np.random.randint(0, 68, size=image.shape[0])
            for i in range(image.shape[0]):
                x, y = coords[i][random_idxs[i]]
                x, y = int(x.floor()), int(y.floor())
                shift_x_low = max(shifting - x, 0)
                shift_x_high = max(shifting - (image.shape[2] - 1 - x), 0)
                shift_y_low = max(shifting - y, 0)
                shift_y_high = max(shifting - (image.shape[3] - 1 - y), 0)

                image[i][:, x-shifting+shift_x_low-shift_x_high:x+shifting+1-shift_x_high+shift_x_low, y-shifting+shift_y_low-shift_y_high:y+shifting+1-shift_y_high+shift_y_low] = white
            

            # Random binary mask
            """
            image = torch.permute(image, (0, 2, 3, 1))
            binary_mask =  np.random.choice([True, False], size=(image.shape[0], image.shape[1], image.shape[2]), p=[0.2, 0.8])
            for i in range(image.shape[0]):
                image[i][binary_mask[i]] = white[binary_mask[i]]
            
            image = torch.permute(image, (0, 3, 1, 2))
            """

            image, coords = image.cuda(), coords.cuda()

            output = model(image)

            loss = criterion(output, coords) + centerLoss(output, coords)
            
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    model.eval()
    NME_loss = 0.
    for image, coords in tqdm(val_loader):
        
        image, coords = image.cuda(), coords.cuda()

        with torch.no_grad():
            output = model(image)
            loss = criterion(output, coords) + centerLoss(output, coords)
            val_loss += loss.item()

            NME = evaluation(output, coords)
            NME_loss += NME.item()

        torch.save(model.state_dict(), os.path.join(save_path, "last_model.pth"))
    
    train_loss /= len(train_set)
    val_loss /= len(val_set)
    NME_loss /= len(val_set)

    log_content = f'Epoch: {epoch+1}/{max_epoch}, Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}, NME: {NME_loss*100:.2f}%'
    print(log_content)

    with open(os.path.join(save_path,  "log.txt"), "a") as log_file:
        log_file.write(log_content)

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
plt.show()