from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import torch
from matplotlib import pyplot

class RMapDataset(Dataset):

    def __init__(self, root, set, obj_name, kpt_num, transform):
        self.root = root
        self.set = set
        self.transform = transform
        self.obj_name = obj_name


        self._imgpath = os.path.join(self.root, 'LINEMOD', self.obj_name, 'JPEGImages', '%s.jpg')
        self._radialpath = os.path.join(self.root, 'LINEMOD', self.obj_name, 'Out_pt'+kpt_num+'_dm', '%s.npy')
        self._imgsetpath = os.path.join(self.root,'LINEMOD', self.obj_name, 'Split', '%s.txt')
        with open(self._imgsetpath % self.set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip('\n') for x in self.ids]

    def __getitem__(self, item):
        img_id = self.ids[item]

        target = np.load(self._radialpath % img_id)
        img = Image.open(self._imgpath % img_id).convert('RGB')

        if self.transform is not None:
            img_torch, target_torch, sem_target_torch = self.transform(img, target)

        return img_torch, target_torch, sem_target_torch

    def __len__(self):
        return len(self.ids)