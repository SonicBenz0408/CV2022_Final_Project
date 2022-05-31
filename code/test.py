from PIL import Image
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

noise_size = 41
white = torch.FloatTensor([[[2.2489]], [[2.4286]], [[2.6400]]]).repeat(1, noise_size, noise_size)

d = Image.open("./data/aflw_val/image01205.jpg")

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

e = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

img = e(d)
x, y = 300, 350

img[:, x-noise_size//2:x+noise_size//2+1, y-noise_size//2:y+noise_size//2+1] = white
img = img.numpy()

plt.imshow(np.transpose(img, (1, 2, 0)))
plt.show()