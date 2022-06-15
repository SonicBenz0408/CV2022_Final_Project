import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
#import pickle5 as pickle
from PIL import Image
import os
from torchvision.transforms.functional import rotate, hflip, pad, crop
import matplotlib.pyplot as plt
from rotation import rotate_coord

class Img_Dataset_train(Dataset):
    def __init__(self, data_path, transforms, have_anno=True):
        
        if have_anno:
            with open(os.path.join(data_path, "annot.pkl"), "rb") as f:
                annot = pickle.load(f)
                self.names, self.feats= annot
        else:
            self.names = os.listdir(data_path)
            self.feats = [[0.]] * len(self.names) 

        self.imgs = []
        for name in self.names:
            img = os.path.join(data_path, name)
            self.imgs.append(img)

        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = Image.open(self.imgs[index])
        feat = torch.FloatTensor(self.feats[index])

        # flip
        hflip_n = np.random.choice([True, False], size=1, p=[0.5, 0.5])

        if hflip_n:
            img = hflip(img)
            feat[:, 0] = 383 - feat[:, 0]
            feat = torch.cat((torch.flip(feat[0:17, :], [0]), torch.flip(feat[17:27, :], [0]), feat[27:31, :], torch.flip(feat[31:36, :], [0]), torch.flip(feat[42:46, :], [0]), torch.flip(feat[46:48, :], [0]), torch.flip(feat[36:40, :], [0]), torch.flip(feat[40:42, :], [0]), torch.flip(feat[48:55, :], [0]), torch.flip(feat[55:60, :], [0]), torch.flip(feat[60:65, :], [0]), torch.flip(feat[65:68, :], [0])))
        
        img = self.transforms(img)        
            
        return img, feat

class Img_Dataset_sim(Dataset):
    def __init__(self, data_path, transforms, have_anno=True):
        
        if have_anno:
            with open(os.path.join(data_path, "annot.pkl"), "rb") as f:
                annot = pickle.load(f)
                self.names, self.feats= annot
        else:
            self.names = os.listdir(data_path)
            self.feats = [[0.]] * len(self.names) 

        self.imgs = []
        for name in self.names:
            img = os.path.join(data_path, name)
            self.imgs.append(img)

        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = Image.open(self.imgs[index])
        feat = torch.FloatTensor(self.feats[index])

        # flip
        hflip_n = np.random.choice([True, False], size=1, p=[0.5, 0.5])

        if hflip_n:
            img = hflip(img)
            feat[:, 0] = 383 - feat[:, 0]
            feat = torch.cat((torch.flip(feat[0:17, :], [0]), torch.flip(feat[17:27, :], [0]), feat[27:31, :], torch.flip(feat[31:36, :], [0]), torch.flip(feat[42:46, :], [0]), torch.flip(feat[46:48, :], [0]), torch.flip(feat[36:40, :], [0]), torch.flip(feat[40:42, :], [0]), torch.flip(feat[48:55, :], [0]), torch.flip(feat[55:60, :], [0]), torch.flip(feat[60:65, :], [0]), torch.flip(feat[65:68, :], [0])))
        
        img = self.transforms(img)

        angle = 90 * (2 * np.random.rand() - 1)
        img = rotate(img, angle)
        feat = rotate_coord(feat, angle)

        new_image = img.unsqueeze(0)
        new_coord = feat.unsqueeze(0)
        for j in range(1, 4):
            new_image = torch.cat((new_image, rotate(img, 90*j).unsqueeze(0)))
            new_coord = torch.cat((new_coord, rotate_coord(feat, 90*j).unsqueeze(0)))
        return new_image, new_coord

class Img_Dataset(Dataset):
    def __init__(self, data_path, transforms, have_anno=True):
        
        if have_anno:
            with open(os.path.join(data_path, "annot.pkl"), "rb") as f:
                annot = pickle.load(f)
                self.names, self.feats= annot
        else:
            self.names = os.listdir(data_path)
            self.feats = [[0.]] * len(self.names) 

        self.imgs = []
        for name in self.names:
            img = os.path.join(data_path, name)
            self.imgs.append(img)

        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = Image.open(self.imgs[index])
        img = self.transforms(img)
        feat = torch.FloatTensor(self.feats[index])
        
        return img, feat