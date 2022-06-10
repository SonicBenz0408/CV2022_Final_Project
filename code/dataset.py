import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
#import pickle5 as pickle
from PIL import Image
import os

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
        img = self.transforms(img)
        feat = torch.FloatTensor(self.feats[index])
        
        return img, feat

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
        img = self.transforms(img)
        feat = torch.FloatTensor(self.feats[index])
        
        return img