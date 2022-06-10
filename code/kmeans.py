import os
import random
import torch
from torchvision import transforms
import numpy as np
from dataset import Img_Dataset_train
from dataset import Img_Dataset_clean
from sklearn.cluster import KMeans
import cv2
import time

train_path = "./data/synthetics_train"
img_w, img_h = 384, 384

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tfms = transforms.Compose([
    transforms.Resize((img_h//2, img_w//2)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
train_set = Img_Dataset_clean(train_path, train_tfms)
print("Dataset complete!")

start = time.time()
# Anchor Generation
k = 10
imgs = []
for i in range(len(train_set) // 100):
# for i in range(100):
    img = train_set[i*100].swapaxes(0, 1).swapaxes(1, 2)
    img = img.reshape((3* img_h * img_w // 4))
    img.unsqueeze_(1)
    # print(img.shape)
    imgs.append(img)
    if i%1000 == 0:
        print(i)
imgs = torch.cat(imgs, dim=1).swapaxes(0, 1)
# print(imgs)
print(imgs.shape)
kmeans_tool = KMeans(n_clusters=k).fit(imgs)
clusters = np.reshape(kmeans_tool.cluster_centers_, (k, img_h//2, img_w//2, 3))
for i in range(len(clusters)):
    cv2.imwrite('cluster_'+str(i)+'.jpg', clusters[i])
end = time.time()
print('time =', end - start)
