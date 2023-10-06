from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import torch
from matplotlib import pyplot
import h5py
import open3d as o3d

def read_depth(path):
    if (path[-3:] == 'dpt'):
        with open(path) as f:
            h,w = np.fromfile(f,dtype=np.uint32,count=2)
            data = np.fromfile(f,dtype=np.uint16,count=w*h)
            depth = data.reshape((h,w))
    else:
        depth = np.asarray(Image.open(path)).copy()
    return depth

class RMapDataset(Dataset):

    def __init__(self, root, dname, set, obj_name, kpt_num, transform):
        self.root = root
        self.set = set
        self.transform = transform
        self.obj_name = obj_name
        self.dname = dname
        self.kpt_num = kpt_num


        if self.dname == 'lm':
            self._imgpath = os.path.join(self.root, 'LINEMOD', self.obj_name, 'JPEGImages', '%s.jpg')
            self._radialpath = os.path.join(self.root, 'LINEMOD_ORIG', self.obj_name, 'Out_pt'+kpt_num+'_dm', '%s.npy')
            self._depthpath = os.path.join(self.root, 'LINEMOD_ORIG', self.obj_name, 'data', 'depth%s.dpt')
            self._maskpath = os.path.join(self.root, 'LINEMOD', self.obj_name, 'mask', '%s.png')
            self._gtposepath = os.path.join(self.root, 'LINEMOD', self.obj_name, 'pose', 'pose%s.npy')
            self._imgsetpath = os.path.join(self.root,'LINEMOD', self.obj_name, 'Split', '%s.txt')

            # load ply 
            #print(self.kpt)        
            cad_model_mm = o3d.io.read_point_cloud(os.path.join(self.root, 'LINEMOD_ORIG', self.obj_name, 'mesh.ply'))
            cad_model_points_m = np.asarray(cad_model_mm.points)/1000
            if os.path.isfile(os.path.join(self.root,'LINEMOD',self.obj_name,'Outside9.npy')):
                self.kpt = np.load(os.path.join(self.root,'LINEMOD',self.obj_name,'Outside9.npy'))
            else:
                print("No kpt file found, generating kpts...")
                BBox = cad_model_mm.get_oriented_bounding_box()
                bboxcorners=np.asarray((BBox.get_box_points()))
                self.kpt = bboxcorners*2
                np.save(os.path.join(self.root,'LINEMOD',self.obj_name,'Outside9.npy'), self.kpt)
            self.kpt = self.kpt[int(kpt_num)]
            #print(cad_model_points)
            dsitances = ((cad_model_points_m[:,0]-self.kpt[0])**2
                         +(cad_model_points_m[:,1]-self.kpt[1])**2
                         +(cad_model_points_m[:,2]-self.kpt[2])**2)**0.5
            self.max_radii_dm = dsitances.max()*10
            #print('maximum radial distance: ', self.max_radii_dm)


            #print(self.kpt)
        else:
            #YCB
            self._h5path = os.path.join(self.root, self.obj_name+'.hdf5')
            self._imgsetpath = os.path.join(self.root,self.obj_name, 'Split', '%s.txt')
            #self.h5f = h5py.File(self._h5path, 'r')
        with open(self._imgsetpath % self.set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip('\n') for x in self.ids]

    def __getitem__(self, item):
        img_id = self.ids[item]

        if self.dname == 'lm':
            target = np.load(self._radialpath % img_id)
            depth = read_depth(self._depthpath % str(int(img_id)))
            mask = np.asarray(Image.open((self._maskpath % str(int(img_id)).zfill(4))),dtype=int)[:,:,0]
            gtpose = np.load(self._gtposepath% str(int(img_id)))
            img = Image.open(self._imgpath % img_id).convert('RGB')
        else:
            #print(self._h5path)
            ycbh5f = h5py.File(self._h5path, 'r')
            target = np.array(ycbh5f['3Dradius_pt'+self.kpt_num+'_dm'][img_id])
            #print(target)
            #img = np.array(ycbh5f['JPEGImages'][img_id])
            img = np.array(ycbh5f[img_id])
            ycbh5f.close()
        if self.transform is not None:
            img_torch, target_torch, sem_target_torch = self.transform(img_id, img, target,depth,mask,gtpose,self.kpt)

        return img_torch, target_torch, sem_target_torch

    def __len__(self):
        return len(self.ids)