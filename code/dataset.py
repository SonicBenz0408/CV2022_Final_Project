import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from PIL import Image
import os

class Img_Dataset(Dataset):
    def __init__(self, data_path, transforms):
        
        with open(os.path.join(data_path, "annot.pkl"), "rb") as f:
            annot = pickle.load(f)
            self.names, self.feats= annot
        
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