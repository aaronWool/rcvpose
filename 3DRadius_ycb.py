import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d
import os
from numba import jit,prange
import json
import scipy.io
import h5py

depthGeneration=False

classlist={1:'002_master_chef_can',
           2:'003_cracker_box',
           3:'004_sugar_box',
           4:'005_tomato_soup_can',
           5:'006_mustard_bottle',
           6:'007_tuna_fish_can',
           7:'008_pudding_box',
           8:'009_gelatin_box',
           9:'010_potted_meat_can',
           10:'011_banana',
           11:'019_pitcher_base',
           12:'021_bleach_cleanser',
           13:'024_bowl',
           14:'025_mug',
           15:'035_power_drill',
           16:'036_wood_block',
           17:'037_scissors',
           18:'040_large_marker',
           19:'051_large_clamp',
           20:'052_extra_large_clamp',
           21:'061_foam_brick'}


def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    #pointc->actual scene
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    actual_xyz=xyz
    #np.savetxt("test1.txt",actual_xyz,delimiter=',')
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
    #print(xyz[:, :2])
    #print(xyz[:, 2:])
    xy = xyz[:, :2] / xyz[:, 2:]
    #xy = xyz[:, :2]
    return xy,actual_xyz

def rgbd_to_point_cloud(K, depth):
    vs, us = depth.nonzero()
    zs = depth[vs, us]
    #print(zs.min())
    #print(zs.max())
    ys = ((us - K[0, 2]) * zs) / float(K[0, 0])
    xs = ((vs - K[1, 2]) * zs) / float(K[1, 1])
    pts = np.array([xs, ys, zs]).T
    return pts, vs, us

@jit(nopython=True, parallel=True)   
def fast_for_map(yList, xList, xyz, distance_list, Radius3DMap):
    for i in prange(len(xList)):
        Radius3DMap[yList[i],xList[i]] = distance_list[i]
    return Radius3DMap

@jit(nopython=True, parallel=True)    
def fast_for(xyz_mm,radial_list_mm,VoteMap_3D):   
    factor = (3**0.5)/4 
    for count in prange(xyz_mm.shape[0]):
        xyz = xyz_mm[count]
        radius = radial_list_mm[count]
        for i in prange(VoteMap_3D.shape[0]):
            for j in prange(VoteMap_3D.shape[1]):
                for k in prange(VoteMap_3D.shape[1]):
                    distance = ((i-xyz[0])**2+(j-xyz[1])**2+(k-xyz[2])**2)**0.5
                    if radius - distance < factor and radius - distance>0:
                        VoteMap_3D[i,j,k]+=1
    return VoteMap_3D

@jit(nopython=True,parallel=True)   
def fast_for_map2(pixel_coor, xy, actual_xyz, distance_list, Radius3DMap):
    z_mean = np.mean(actual_xyz[:,2])
    
    for coor in pixel_coor:
        iter_count=0
        z_loc = 0
        pre_z_loc=0
        z_min = 99999999999999999
        for xy_single in xy:
            if(coor[0]==xy_single[1] and coor[1]==xy_single[0]):
                #print(coor)
                #print(xy_single)
                #print(actual_xyz[iter_count,2])
                if(actual_xyz[iter_count,2]<z_min):
                    z_loc = iter_count
                    z_min = actual_xyz[iter_count,2]
                    Radius3DMap[xy[z_loc][1],xy[z_loc][0]]=actual_xyz[z_loc,2]
            iter_count+=1
    return Radius3DMap
    
def Accumulator_3D(xyz, radial_list, pixel_coor):
    #accumulator space unit: mm
    acc_unit = 5
    # unit 5mm 
    xyz_mm = xyz*1000/acc_unit #point cloud is in meter
    #recenter the point cloud
    x_mean_mm = np.mean(xyz_mm[:,0])
    y_mean_mm = np.mean(xyz_mm[:,1])
    z_mean_mm = np.mean(xyz_mm[:,2])
    xyz_mm[:,0] -= x_mean_mm
    xyz_mm[:,1] -= y_mean_mm
    xyz_mm[:,2] -= z_mean_mm
    
    radial_list_mm = radial_list*1000/acc_unit  #radius map is in decimetre for training purpose
    
    xyz_mm_min = xyz_mm.min()
    xyz_mm_max = xyz_mm.max()
    radius_max = radial_list_mm.max()
    
    zero_boundary = int(xyz_mm_min-radius_max)+1
    #print("debug zero boundary: ",zero_boundary)
    
    if(zero_boundary<0):
        xyz_mm = xyz_mm-zero_boundary
        #length of 3D vote map 
    length = int(xyz_mm.max())+1
    #print(length)
    
    VoteMap_3D = np.zeros((length,length,length))
    VoteMap_3D = fast_for(xyz_mm,radial_list_mm,VoteMap_3D)
    #fast_for_cuda(xyz_mm,radial_list_mm,VoteMap_3D)
    #np.save("sampleVoteMapGT.npy", VoteMap_3D)
    #os.system("pause")
    #print(VoteMap_3D)
    print(VoteMap_3D.max())
    #os.system("pause")
                        
    center = np.argwhere(VoteMap_3D==VoteMap_3D.max())
    if(zero_boundary<0):
        center = center+zero_boundary
        
    #return to global coordinate
    center[0,0] += x_mean_mm
    center[0,1] += y_mean_mm
    center[0,2] += z_mean_mm
    
    center = center*acc_unit+0.5

    return center


def gen_GT(opts):
    rootdict = opts.root_dataset
    savedict = opts.root_save
    for cycle in os.listdir(rootdict):
        print(cycle)
        if(True):
        #if(int(cycle)>=10):
            for filename in os.listdir(rootdict+cycle+"/"):
                if(os.path.splitext(filename)[1]=='.mat'):
                    #print(filename)
                    sceneInfo = scipy.io.loadmat(rootdict+cycle+"/"+filename)
                    class_idx = sceneInfo['cls_indexes']
                    #poses=sceneInfo['poses']
                    #print(sceneInfo['cls_indexes'])
                    mask=np.asarray(Image.open(rootdict+cycle+"/"+os.path.splitext(filename)[0][0:6]+"-label.png"))
                    depth=np.asarray(Image.open(rootdict+cycle+"/"+os.path.splitext(filename)[0][0:6]+"-depth.png"))
                    depthFactor=sceneInfo['factor_depth']
                    #print(depthFactor)
                    depth = depth/depthFactor
                    #divided by factor
                    #print(depth.shape)
                    cam_k = sceneInfo['intrinsic_matrix']
                    obj_idx=0
                    for classname in class_idx:
                        #print(classname)
                        #print(obj_idx)
                        #os.system("pause")
                        if classname[0] in classlist:
                            single_mask=np.where(mask==classname[0],1,0)
                            #pixel_coor = np.argwhere(single_mask==1)
                            #plt.imshow(single_mask)
                            #plt.show()
                            keypoints = np.load(rootdict + 'models/'+classlist[classname[0]]+"/Outside9.npy")
                            idx_points=0
                            
                            #h5 group
                            h5file = h5py.File(savedict+classlist[classname[0]]+'.hdf5','a')
                            node = "/JPEGImages/"
                            if node not in h5file.keys():
                                JPEGImagesGroup = h5file.create_group("/JPEGImages/")
                            else:
                                JPEGImagesGroup = h5file["/JPEGImages/"]
                            jpgImage = np.asarray(Image.open(rootdict+cycle+"/"+os.path.splitext(filename)[0][0:6]+"-color.png").convert('RGB'))
                            if "/JPEGImages/"+cycle+"_"+os.path.splitext(filename)[0][0:6] not in  h5file.keys():
                                JPEGImagesGroup.create_dataset(cycle+"_"+os.path.splitext(filename)[0][0:6], data=jpgImage,  compression="gzip", compression_opts=9)
    
                            for keypoint in keypoints:
                                node = "/3Dradius_pt"+str(idx_points)+"_dm/"
                                if node not in h5file.keys():
                                    kptGroup = h5file.create_group("/3Dradius_pt"+str(idx_points)+"_dm/")
                                else:
                                    kptGroup = h5file["/3Dradius_pt"+str(idx_points)+"_dm/"]
                                xyz=scipy.io.loadmat(rootdict +"models/"+classlist[classname[0]]+".mat")
                                xyz = xyz['obj'][0][0][1]
                                #xyz=np.transpose(xyz)
                                #xyz = xyz/2
                                #print(xyz)
                                #RT
                                #R=np.transpose(poses[iter_][0:3])
                                #T=poses[iter_][3].reshape(3,1)
                                #RT = np.append(R,T,axis=1)
                                #print(sceneInfo['center'][obj_idx])
                                #print(cam_k)
                                RT = sceneInfo['poses'][:,:,obj_idx]
                                #RTnp.linalg.inv(RT)
                                #print(RT)
                                #print(xyz)
                                #xy, true_xyz = project(xyz,cam_k,RT)
                                #os.system("pause")
                                #print(xy.shape)
                                #print(true_xyz.shape)
                                
                                masked_depth = depth*single_mask
                                xyz,y,x = rgbd_to_point_cloud(cam_k, masked_depth)
                                #np.savetxt("test2.txt",xyz,delimiter=',')
                                dump, transformed_kpoint = project(np.array([keypoint]),cam_k,RT)
                                transformed_kpoint = transformed_kpoint[0]
                                #print('estimated: ',transformed_kpoint)
                                #print(dump)
                                distance_list = ((xyz[:,0]-transformed_kpoint[0])**2+(xyz[:,1]-transformed_kpoint[1])**2+(xyz[:,2]-transformed_kpoint[2])**2)**0.5
                                #distance_list = ((true_xyz[:,0]-transformed_kpoint[0])**2+(true_xyz[:,1]-transformed_kpoint[1])**2+(true_xyz[:,2]-transformed_kpoint[2])**2)**0.5
                                Radius3DMap = np.zeros(mask.shape)
                                Radius3DMap = fast_for_map(y, x, xyz, distance_list, Radius3DMap)
                                #xy = np.around(xy).astype(int)
                                #Radius3DMap = fast_for_map2(pixel_coor,xy,true_xyz,distance_list,Radius3DMap)
                                #np.save(saveRDict+cycle+"_"+os.path.splitext(filename)[0][0:6]+".npy",Radius3DMap*10)
                                #print(saveRDict+cycle+"_"+os.path.splitext(filename)[0][0:6]+".npy")
                                #print("/ycbRCVData/"+classlist[classname[0]]+"/3Dradius_pt"+str(idx_points)+"_dm/"+cycle+"_"+os.path.splitext(filename)[0][0:6])
                                if "/3Dradius_pt"+str(idx_points)+"_dm/"+cycle+"_"+os.path.splitext(filename)[0][0:6] not in  h5file.keys():
                                    kptGroup.create_dataset(cycle+"_"+os.path.splitext(filename)[0][0:6], data = Radius3DMap*10,  compression="gzip", compression_opts=9)
                                #os.system("pause")
    
                                #print(xyz)
                                #print(distance_list)
                                
                                #pixel_coor = np.where(single_mask==1)
                                #print(pixel_coor)
                                #center_mm_s = Accumulator_3D(xyz, distance_list, pixel_coor)
                                #print('voted: ',center_mm_s)
                                #os.system("pause")
                                #print(xy)
                                #testMask=np.zeros(mask.shape)
                                ##print(xy)
                                #for coor in xy:
                                #    #print(coor)
                                #    if int(coor[1])<480 and int(coor[1])>=0 and int(coor[0])<640 and int(coor[0])>=0:
                                #        testMask[int(coor[1]),int(coor[0])]=1
                                #plt.imshow(masked_depth-Radius3DMap)
                                #plt.show()
                                #print(xyz)
                                idx_points+=1
                            h5file.close()
                        obj_idx+=1
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dataset',
                    type=str,
                    default='Datasets/ycb/')
    parser.add_argument('--root_save',
                    type=str,
                    default='Datasets/ycb/')   
    opts = parser.parse_args()   