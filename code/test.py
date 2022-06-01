from PIL import Image
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from dataset import Img_Dataset
from torchvision.transforms.functional import rotate
from rotation import rotate_coord
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
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_set = Img_Dataset(train_path, val_tfms)

angle = float(np.random.rand(1)[0])
img = train_set[0][0]
img_r = rotate(img, angle)
coord = train_set[0][1]
coord_r = rotate_coord(coord, angle)
plt.figure(0)
plt.imshow(np.transpose(img, (1, 2, 0)))
plt.scatter(coord[:, 0], coord[:, 1], c="g", s=2)
plt.figure(1)
plt.imshow(np.transpose(img_r, (1, 2, 0)))
plt.scatter(coord_r[:, 0], coord_r[:, 1], c="g", s=2)
plt.show()