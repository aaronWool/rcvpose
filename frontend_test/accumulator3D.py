import numpy as np
import time
from numba import jit,njit,cuda
import os
from numba import prange
import math



@jit(nopython=True,parallel=True)
#@jit(parallel=True)     
def fast_for(xyz_mm,radial_list_mm,VoteMap_3D):  
    factor = (3**0.5)/4
    for count in prange(xyz_mm.shape[0]):
        xyz = xyz_mm[count]
        radius = radial_list_mm[count]
        radius = int(np.around(radial_list_mm[count]))
        shape = VoteMap_3D.shape
        for i in prange(VoteMap_3D.shape[0]):
            for j in prange(VoteMap_3D.shape[1]):
                for k in prange(VoteMap_3D.shape[2]):
                    distance = ((i-xyz[0])**2+(j-xyz[1])**2+(k-xyz[2])**2)**0.5
                    if radius - distance < factor and radius - distance>0:
                        VoteMap_3D[i,j,k]+=1
        
    return VoteMap_3D



def Accumulator_3D(xyz, radial_list):
    acc_unit = 5
    # unit 5mm 
    xyz_mm = xyz*1000/acc_unit #point cloud is in meter

    #print(xyz_mm)
    
    #recenter the point cloud
    x_mean_mm = np.mean(xyz_mm[:,0])
    y_mean_mm = np.mean(xyz_mm[:,1])
    z_mean_mm = np.mean(xyz_mm[:,2])
    xyz_mm[:,0] -= x_mean_mm
    xyz_mm[:,1] -= y_mean_mm
    xyz_mm[:,2] -= z_mean_mm
    
    radial_list_mm = radial_list*100/acc_unit  #radius map is in decimetre for training purpose
    
    xyz_mm_min = xyz_mm.min()
    xyz_mm_max = xyz_mm.max()
    radius_max = radial_list_mm.max()
    
    zero_boundary = int(xyz_mm_min-radius_max)+1
    
    if(zero_boundary<0):
        xyz_mm -= zero_boundary
        #length of 3D vote map 
    length = int(xyz_mm.max())
    
    VoteMap_3D = np.zeros((length+int(radius_max),length+int(radius_max),length+int(radius_max)))
    tic = time.perf_counter()
    VoteMap_3D = fast_for(xyz_mm,radial_list_mm,VoteMap_3D)
    toc = time.perf_counter()
                        
    center = np.argwhere(VoteMap_3D==VoteMap_3D.max())
   # print("debug center raw: ",center)
    center = center.astype("float64")
    if(zero_boundary<0):
        center = center+zero_boundary
        
    #return to global coordinate
    center[0,0] = (center[0,0]+x_mean_mm+0.5)*acc_unit
    center[0,1] = (center[0,1]+y_mean_mm+0.5)*acc_unit
    center[0,2] = (center[0,2]+z_mean_mm+0.5)*acc_unit
    
    #center = center*acc_unit+((3**0.5)/2)

    return center
