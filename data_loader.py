# from torchvision.datasets import VOCSegmentation
from rmap_dataset import RMapDataset
from torch.utils import data
import torch
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange

linemod_K = np.array([[572.4114, 0., 325.2611],
                  [0., 573.57043, 242.04899],
                  [0., 0., 1.]])

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    #pointc->actual scene
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    actual_xyz=xyz
    #np.set_printoptions(threshold=np.inf)
    #xyz = xyz[np.where(xyz[:,2]>=1)]
    #distance_list = distance_list[np.where(xyz[:,2]>=1)]
    #print(xyz)
    #np.savetxt('GT.txt', actual_xyz*1000, delimiter=' ') 
    #os.system("pause")
    #scene->image space
    xyz = np.dot(xyz, K.T)
    #np.set_printoptions(threshold=np.inf)
    #print(xyz)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy,actual_xyz

def rgbd_to_point_cloud(K, depth):
    vs, us = depth.nonzero()
    zs = depth[vs, us]
    #print(zs.min())
    #print(zs.max())
    xs = ((us - K[0, 2]) * zs) / float(K[0, 0])
    ys = ((vs - K[1, 2]) * zs) / float(K[1, 1])
    pts = np.array([xs, ys, zs]).T
    return pts, vs, us

@jit(nopython=True, parallel=True)   
def fast_for_map(yList, xList, xyz, distance_list, Radius3DMap):
    for i in prange(len(xList)):
        Radius3DMap[yList[i],xList[i]] = distance_list[i]
    return Radius3DMap


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

    def transform(self, img_id, img, depth,mask,gtpose,kpt):
        #print(img_id)
        #generate gt radius label
        Radius3DMap = np.zeros(mask.shape)
        pixel_coor = np.argwhere(mask==255)
        depth[np.where(mask==0)] = 0
        xyz_mm,y,x = rgbd_to_point_cloud(linemod_K, depth)
        xyz=xyz_mm/1000
        #print(xyz)
        #print(gtpose.shape)
        gtpose_mm = gtpose.copy()
        gtpose_mm[:,3:] = gtpose[:,3:]*1000
        #print(linemod_K_m)
        kpt_mm = kpt*1000
        #print(kpt_mm)
        dump, transformed_kpoint = project(np.array([kpt_mm]),linemod_K,gtpose_mm)
        #print(transformed_kpoint)
        transformed_kpoint=transformed_kpoint[0]/1000
        distance_list = ((xyz[:,0]-transformed_kpoint[0])**2+(xyz[:,1]-transformed_kpoint[1])**2+(xyz[:,2]-transformed_kpoint[2])**2)**0.5
        Radius3DMap = fast_for_map(y, x, xyz, distance_list, Radius3DMap)
        img = np.array(img, dtype=np.float64)
        img /= 255.
        lbl = np.array(Radius3DMap, dtype=np.float64)
        lbl = lbl*10
        lbl = np.where(lbl>self.max_radii_dm,0,lbl)
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
    modes = ['val', 'val']
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
