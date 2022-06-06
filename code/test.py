from PIL import Image
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from dataset import Img_Dataset
from torchvision.transforms.functional import rotate
from rotation import rotate_coord
from torchvision.transforms import functional as F
"""
noise_size = 41
white = torch.FloatTensor([[[2.2489]], [[2.4286]], [[2.6400]]]).repeat(1, noise_size, noise_size)

d = Image.open("./data/aflw_val/image00182.jpg")

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

e = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

img = e(d)
print(img)
x, y = 300, 350

img[:, x-noise_size//2:x+noise_size//2+1, y-noise_size//2:y+noise_size//2+1] = white
img = img.numpy()
"""

train_path = "./data/synthetics_train"
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

val_tfms = transforms.Compose([
    #transforms.Pad(100),
    #transforms.RandomCrop((384, 384)),
    #transforms.ToTensor(),
    #transforms.Normalize(mean, std)
])

train_set = Img_Dataset(train_path, val_tfms)
for i in range(100):
    img = train_set[i][0]
    coord = train_set[i][1]
    pad_size = 125
    img = F.pad(img, pad_size)
    top = np.random.random() * (2*pad_size)
    left = np.random.random() * (2*pad_size)
    coord[:, 0] += 150 - left
    coord[:, 1] += 150 - top
    img = F.crop(img, top, left, 384, 384)
    plt.figure()
    plt.imshow(img)
    plt.scatter(coord[:, 0], coord[:, 1], c="g", s=2)
    plt.show()
"""
angle = float(np.random.rand(1)[0])
img = train_set[0][0]
coord = train_set[0][1]
coords = train_set[0][1]
image = F.hflip(img)
coords[:, 0] = 383 - coord[:, 0]
plt.figure(0)
plt.imshow(np.transpose(img, (1, 2, 0)))
plt.scatter(coord[:, 0], coord[:, 1], c="g", s=2)
plt.figure(1)
plt.imshow(np.transpose(image, (1, 2, 0)))
plt.scatter(coords[:, 0], coords[:, 1], c="g", s=2)
plt.show()
"""