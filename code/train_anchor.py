import os
import random
import torch.nn as nn
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from dataset import Img_Dataset_train
from dataset import Img_Dataset
from pyramid import PyramidNet, AnchorPyramidNet
from loss import WingLoss, NMELoss, WeightedL2Loss, CenterLoss, RegressionLoss, ConfidenceLoss
from torchvision.transforms.functional import rotate, hflip, pad, crop, affine
from rotation import rotate_coord
from sklearn.cluster import KMeans

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
    transforms.ColorJitter(brightness=0.2, saturation=0.2, hue=0.2),
    #transforms.GaussianBlur(5),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_set = Img_Dataset_train(train_path, train_tfms)
val_set = Img_Dataset(val_path, val_tfms)

print("Dataset complete!")

# Anchor Generation
k = 3
all_feats = np.reshape(np.array(train_set.feats), (len(train_set.feats), 136))
kmeans_tool = KMeans(n_clusters=k).fit(all_feats)
clusters = torch.FloatTensor(np.reshape(kmeans_tool.cluster_centers_, (k, 68, 2)))

#for i in range(k):
#    plt.figure(i, figsize=(4, 4))
#    plt.xlim(0, 383)
#    plt.ylim(383, 0)
#    plt.scatter(clusters[i, :, 0], clusters[i, :, 1], c="r", s=2)
#plt.show()

anchors = []
for i in range(k):
    for j in range(6):
        anchors.append(torch.unsqueeze(rotate_coord(clusters[i], 60 * j), 0))

anchors = torch.cat(anchors)
# Hyperparameters
batch_size = 32
learning_rate = 0.001
max_epoch = 40
noise_epoch = 1
crop_epoch = 3
noise_size = 41
shifting = noise_size // 2
pad_size = 100

center_gamma = 0.05
conf_gamma = 0.5
c_th = 0.6

weights = [1.] * 27 + [20] * 9 + [1] * 24 + [20] * 8
weights = torch.FloatTensor(weights)
#print(weights)

#black = torch.FloatTensor([[[-2.1179]], [[-2.0357]], [[-1.8044]]]).repeat(1, noise_size, noise_size)
white = torch.FloatTensor([[[2.2489]], [[2.4286]], [[2.6400]]]).repeat(1, noise_size, noise_size)
#white = torch.Tensor([2.2489, 2.4286, 2.6400]).repeat(384, 384, 1)
save_path = "./log/MobileNetv2_32_anchor24"

os.makedirs(save_path, exist_ok=True)
# Data loader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

print("Dataloader complete!")

# Preparation
#model = PyramidNet().cuda()
model = AnchorPyramidNet(anchor_num=18).cuda()
optimizer = torch.optim.Adam([{"params":model.parameters(), "initial_lr": learning_rate}], lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, len(train_loader)*10, 0.5)

#criterion = WingLoss()
#centerLoss = CenterLoss()
#criterion = WeightedL2Loss(weights=weights)
#criterion = NMELoss()

regLoss = RegressionLoss(anchors)
confLoss = ConfidenceLoss(anchors)
evaluation = NMELoss()

best_NME = 100.
train_loss_curve, val_loss_curve = [], []
NME_curve = []
print("Training Start!")

count = 0
for epoch in range(max_epoch):

    model.train()

    wing_loss, center_loss = 0., 0.
    train_loss, val_loss = 0., 0.
    for image, coords in tqdm(train_loader):
        #angle_list = np.random.randn(len(image)) * 40
        
        if epoch > crop_epoch:
            crop_list = np.random.choice([True, False], size=len(image), p=[0.1, 0.9])
            for i in range(len(image)):
                #plt.figure(0)
                #plt.imshow(np.transpose(image[i], (1, 2, 0)))
                #plt.scatter(coords[i, :, 0], coords[i, :, 1], c="g", s=2)
                
                # crop
                if crop_list[i]:
                    h_shift = int(np.ceil(np.random.random() * (pad_size)))
                    v_shift = int(np.ceil(np.random.random() * (pad_size)))
                    coords[i, :, 0] += h_shift
                    coords[i, :, 1] += v_shift
                    image[i] = affine(image[i], angle=0, translate=(h_shift, v_shift), scale=1.0, shear=(0.0, 0.0))

        angle_list = 180 * (2 * np.random.rand(len(image)) - 1)
        for i in range(len(image)):
            #plt.figure(0)
            #plt.imshow(np.transpose(image[i], (1, 2, 0)))
            #plt.scatter(coords[i, :, 0], coords[i, :, 1], c="g", s=2)

            image[i] = rotate(image[i], angle_list[i])
            coords[i] = rotate_coord(coords[i], angle_list[i])

            #plt.figure(1)
            #plt.imshow(np.transpose(image[i], (1, 2, 0)))
            #plt.scatter(coords[i, :, 0], coords[i, :, 1], c="g", s=2)
            #plt.show()

        if epoch > noise_epoch:
            random_idxs = np.random.randint(0, 68, size=image.shape[0])
            for i in range(image.shape[0]):
                x, y = coords[i][random_idxs[i]]
                x, y = int(x.floor()), int(y.floor())
                shift_x_low = max(shifting - x, 0)
                shift_x_high = max(shifting - (383 - x), 0)
                shift_y_low = max(shifting - y, 0)
                shift_y_high = max(shifting - (383 - y), 0)

                image[i][:, y-shifting+shift_y_low-shift_y_high:y+shifting+1-shift_y_high+shift_y_low, x-shifting+shift_x_low-shift_x_high:x+shifting+1-shift_x_high+shift_x_low] = white

        image, coords = image.cuda(), coords.cuda()
        
        # (N, A, 68, 2), (N, A)
        r_output, c_output = model(image)
        if count % 30 == 0:
            print(c_output[0])
        loss = regLoss(r_output, coords, c_output) + conf_gamma * confLoss(coords, c_output)
        
        """
        output = model(image)

        wing_l = criterion(output, coords)
        center_l = center_gamma * centerLoss(output, coords)
        loss =  wing_l + center_l
        
        train_loss += loss.item()
        wing_loss += wing_l
        center_loss += center_l
        """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        count += 1

    model.eval()
    NME_loss = 0.
    for image, coords in tqdm(val_loader):
        
        image, coords = image.cuda(), coords.cuda()

        with torch.no_grad():
            
            r_output, c_output = model(image)

            loss = regLoss(r_output, coords, c_output) + conf_gamma * confLoss(coords, c_output)
            
            c_output[c_output < c_th] = 0.
            c_sum = torch.sum(c_output, dim=1)

            c_output = c_output.unsqueeze(dim=2).unsqueeze(dim=3).repeat(1, 1, 68, 2)
            
            output = torch.sum(c_output * r_output, dim=1) / c_sum
            """
            output = model(image)

            loss = criterion(output, coords) + center_gamma * centerLoss(output, coords)
            """
            val_loss += loss.item()

            NME = evaluation(output, coords)
            NME_loss += NME.item()
    
    
    train_loss /= len(train_loader)
    #wing_loss /= len(train_loader)
    #center_loss /= len(train_loader)
    val_loss /= len(val_loader)
    NME_loss /= len(val_loader)

    

    log_content = f'Epoch: {epoch+1}/{max_epoch}, Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}, NME: {NME_loss*100:.3f}%'
    print(log_content)
    print(f'Wing loss: {wing_loss:.4f}, Center loss: {center_loss:.4f}')
    if NME_loss < best_NME:
        best_NME = NME_loss
        torch.save(model.state_dict(), os.path.join(save_path, "last_model.pth"))
        print("save best model.")

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