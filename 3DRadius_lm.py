import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d
import os
from numba import jit, prange

linemod_cls_names = ['ape','benchvise','cam','can','cat','driller','duck','eggbox','glue','holepuncher','iron','lamp','phone']

linemod_K = np.array([[572.4114, 0., 325.2611],
                  [0., 573.57043, 242.04899],
                  [0., 0., 1.]])

depthGeneration = False

linemod_path = "datasets/LINEMOD/"
original_linemod_path = "datasets/LINEMOD_ORIG/"

#IO function from PVNet               
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

#IO function from BOP toolbox 
def linemod_pose(path, i):
    """
    read a 3x3 rotation and 3x1 translation.
    
    can be done with np.loadtxt, but this is way faster
    @return R, t in [cm]
    """
    R = open("{}/data/rot{}.rot".format(path, i))
    R.readline()
    R = np.float32(R.read().split()).reshape((3, 3))

    t = open("{}/data/tra{}.tra".format(path, i))
    t.readline()
    t = np.float32(t.read().split())
    
    return R, t

#IO function from BOP toolbox 
def read_depth(path):
    if (path[-3:] == 'dpt'):
        with open(path) as f:
            h,w = np.fromfile(f,dtype=np.uint32,count=2)
            data = np.fromfile(f,dtype=np.uint16,count=w*h)
            depth = data.reshape((h,w))
    else:
        depth = np.asarray(Image.open(path)).copy()
    return depth
    
@jit(nopython=True, parallel=True)   
def fast_for(pixel_coor, xy, actual_xyz, distance_list, Radius3DMap):
    z_mean = np.mean(actual_xyz[:,2])
    for coor in pixel_coor:
        iter_count=0
        z_loc = 0
        z_min = 99999999999999999
        for xy_single in xy:
            if(coor[0]==xy_single[1] and coor[1]==xy_single[0]):
                #print(coor)
                #print(xy_single)
                #print(actual_xyz[iter_count,2])
                if(actual_xyz[iter_count,2]<z_min):
                    z_loc = iter_count
                    z_min = actual_xyz[iter_count,2]
                    #Radius3DMap[xy[z_loc][1],xy[z_loc][0]]=distance_list[z_loc]
            iter_count+=1
        
        
        if(z_min<=z_mean):
            if depthGeneration:
                Radius3DMap[xy[z_loc][1],xy[z_loc][0]]=actual_xyz[z_loc,2]
                pre_z_loc = z_loc
            else:
                Radius3DMap[xy[z_loc][1],xy[z_loc][0]]=distance_list[z_loc]
                pre_z_loc = z_loc
        else:
            if depthGeneration:
                Radius3DMap[xy[z_loc][1],xy[z_loc][0]]=actual_xyz[pre_z_loc,2]
            else:
                Radius3DMap[xy[z_loc][1],xy[z_loc][0]]=distance_list[pre_z_loc]
    return Radius3DMap
    
z_min = 999999999999999999
z_max = 0
depth_list=[]



if __name__=='__main__':
    for class_name in linemod_cls_names:
        print(class_name)
        
        pcd_load = o3d.io.read_point_cloud(linemod_path+class_name+"/"+class_name+".ply")
    
        xyz_load = np.asarray(pcd_load.points)
        print(xyz_load)
        
        #x_mean = np.mean(xyz_load[:,0])
        #y_mean = np.mean(xyz_load[:,1])
        #z_mean = np.mean(xyz_load[:,2])

        keypoints=np.load(linemod_path+class_name+"/"+"Outside9.npy")
        points_count = 1
        
        for keypoint in keypoints:
            #keypoint = keypoints[1]
            print(keypoint)
        
            x_mean = keypoint[0]   
            y_mean = keypoint[1] 
            z_mean = keypoint[2]
            
            rootDict = original_linemod_path+class_name+"/" 
            GTDepthPath = rootDict+'FakeDepth/'
            if depthGeneration:
                saveDict = original_linemod_path+class_name+"/FakeDepth/"
            else:
                saveDict = original_linemod_path+class_name+"/Out_pt"+str(points_count)+"_dm/"    
            if(os.path.exists(saveDict)==False):
                os.mkdir(saveDict)
            points_count+=1
            iter_count = 0
            dataDict = rootDict + "data/"
            for filename in os.listdir(dataDict):
                if filename.endswith(".dpt"):
                    #and os.path.exists(saveDict+os.path.splitext(filename)[0][5:].zfill(6)+'.npy')==False
                    print(filename)
                    #depth = np.load(GTDepthPath+os.path.splitext(filename)[0][5:].zfill(6)+'.npy')*1000
                    realdepth = read_depth(dataDict+filename)
                    mask = np.asarray(Image.open(linemod_path+class_name+"/mask/"+os.path.splitext(filename)[0][5:].zfill(4)+".png"), dtype=int)
                    mask = mask[:,:,0]        
                    #depth[np.where(mask==0)] = 0
                    realdepth[np.where(mask==0)] = 0
                    #plt.imshow(realdepth-depth*1000)
                    #plt.show()
                    Radius3DMap = np.zeros(mask.shape)
                    RT = np.load(linemod_path+class_name+"/pose/pose"+os.path.splitext(filename)[0][5:]+".npy")
                    print(RT)
                    print(linemod_pose(rootDict,os.path.splitext(filename)[0][5:]))
                    pixel_coor = np.argwhere(mask==255)
                    xyz,y,x = rgbd_to_point_cloud(linemod_K, realdepth)
                    print(xyz)
                    print(RT)
                    dump, transformed_kpoint = project(np.array([keypoint]),linemod_K,RT)
                    transformed_kpoint = transformed_kpoint[0]*1000
                    print(transformed_kpoint)
                    distance_list = ((xyz[:,0]-transformed_kpoint[0])**2+(xyz[:,1]-transformed_kpoint[1])**2+(xyz[:,2]-transformed_kpoint[2])**2)**0.5
                    Radius3DMap = fast_for_map(y, x, xyz, distance_list, Radius3DMap)
                    #xy, actual_xyz=project(xyz_load,linemod_K,RT)
                    #dump,kpt = project(np.expand_dims(keypoint,axis=0),linemod_K,RT)
                    #distance_list = ((actual_xyz[:,0]-kpt[0,0])**2+(actual_xyz[:,1]-kpt[0,1])**2+(actual_xyz[:,2]-kpt[0,2])**2)**0.5
                    #xy = np.around(xy).astype(int)
                    #xy = xy.astype(int)
                    #xy = xy[np.where(xy[:,0]>0)]
                    #xy = xy[np.where(xy[:,1]>0)]
                    #distance_list = distance_list[np.where(xy[:,0]>0)]
                    #distance_list = distance_list[np.where(xy[:,1]>0)]
                    #Radius3DMap = fast_for(pixel_coor, xy, actual_xyz, distance_list, Radius3DMap)
                    #if(iter_count==0):
                    #    depth_list = Radius3DMap[Radius3DMap.nonzero()]
                    #else:
                    #    depth_list = np.append(depth_list,Radius3DMap[Radius3DMap.nonzero()])
                    iter_count+=1
                    #print(Radius3DMap[Radius3DMap.nonzero()])
                    plt.imshow(Radius3DMap)
                    plt.show()
                    mean = 0.84241277810665
                    std = 0.12497967663932731
                    #Radius3DMap[Radius3DMap.nonzero()] = (Radius3DMap[Radius3DMap.nonzero()]-mean)/std
                    #depth = depth/1000
                    #Radius3DMap[Radius3DMap.nonzero()] = (Radius3DMap[Radius3DMap.nonzero()] - min(depth[depth.nonzero()])) / (max(depth[depth.nonzero()])-min(depth[depth.nonzero()]))
                    #plt.imshow(Radius3DMap)
                    #plt.show()
                    #plt.imshow(np.where(Radius3DMap*100>0,1,0))
                    #plt.show()
                    if depthGeneration:
                        np.save(saveDict+os.path.splitext(filename)[0][5:].zfill(6)+'.npy',Radius3DMap)
                    else:
                        np.save(saveDict+os.path.splitext(filename)[0][5:].zfill(6)+'.npy',Radius3DMap*10)
                    #Radius3DMap[xy[:,1],xy[:,0]]=distance_list
                    #display= depth-Radius3DMap*10
                    #if(Radius3DMap.min()<z_min):
                    #    z_min = Radius3DMap.min()
                    if(Radius3DMap.max()>z_max):
                        z_max = Radius3DMap.max()
                    #Radius3DMap = (Radius3DMap - Radius3DMap[Radius3DMap.nonzero()].min())/(Radius3DMap.max()-Radius3DMap[Radius3DMap.nonzero()].min())
                    #Radius3DMap = np.where(Radius3DMap!=0,Radius3DMap*1000,0)
                    #np.save(os.path.splitext(filename)[0][5:].zfill(6)+'.npy',Radius3DMap)
                    #plt.imshow(Radius3DMap*10)
                    #plt.show()
            if depthGeneration:
                break
            #os.system('pause')
            #print(class_name+" mean: ", np.mean(np.asarray(depth_list)))
            #print(class_name+" std: ", np.std(np.asarray(depth_list)))
            #print("min: ", z_min)
            #print("max: ", z_max)