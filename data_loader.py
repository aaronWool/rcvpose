# from torchvision.datasets import VOCSegmentation
from rmap_dataset import RMapDataset
from torch.utils import data
import torch
import numpy as np
import matplotlib.pyplot as plt


class RData(RMapDataset):

    def __init__(self, root, dname, set='train', obj_name = 'ape', kpt_num = '1'):
        transform = self.transform
        #imageNet mean and std
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.dname = dname
        super().__init__(root,
                        dname,
                        set=set,
                        obj_name = obj_name,
                        kpt_num = kpt_num,
                        transform=transform
                        )

    def transform(self, img, lbl):
        img = np.array(img, dtype=np.float64)
        img /= 255.
        lbl = np.array(lbl, dtype=np.float64)
        if(len(lbl.shape)==2):
          lbl = np.expand_dims(lbl,axis=0)
        img -= self.mean
        img /= self.std
        if img.shape[0] % 2:
            img = img[0:img.shape[0]-1,:]

        if img.shape[1] % 2:
            img = img[:, 0:img.shape[1]-1]

        #print(img.shape)
        sem_lbl = np.where(lbl > 0, 1, -1)

        #filter noise for ycb
        if self.dname != 'lm':
            lbl = np.where(lbl>=10,0,lbl)

        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).float()
        sem_lbl = torch.from_numpy(sem_lbl).float()
        return img, lbl, sem_lbl


    def __len__(self):
        return len(self.ids)


def get_loader(opts):
    from data_loader import RData
    modes = ['train', 'val']
    train_loader = data.DataLoader(RData(opts.root_dataset,
                                        opts.dname,
                                        set=modes[0],
                                        obj_name = opts.class_name,
                                        kpt_num = opts.kpt_num),
                                        batch_size=int(opts.batch_size),
                                        shuffle=True,
                                        num_workers=1)
    val_loader = data.DataLoader(RData(opts.root_dataset,
                                        opts.dname,
                                       set=modes[1],
                                       obj_name = opts.class_name,
                                        kpt_num = opts.kpt_num),
                                       batch_size=int(opts.batch_size),
                                       shuffle=False,
                                       num_workers=1)
    return train_loader, val_loader
