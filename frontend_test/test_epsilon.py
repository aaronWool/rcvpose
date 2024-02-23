import numpy as np
from PIL import Image
import matplotlib.pyplot  as plt
import os
import time
from ransac_vanilla import RANSAC_vanilla
from ransac import RANSAC_3D
from ransac_to_accumulator import RANSAC_Accumulator
import datetime
from accumulator3D import Accumulator_3D
from tqdm import tqdm
import random
import open3d as o3d
import warnings
warnings.filterwarnings("ignore")

lm_cls_names = ['ape', 'benchvise', 'cam', 'can', 'cat', 'duck', 'driller', 'eggbox', 'glue', 'holepuncher','iron','lamp','phone']

linemod_K = np.array([[572.4114, 0., 325.2611],
                  [0., 573.57043, 242.04899],
                  [0., 0., 1.]])

def rgbd_to_point_cloud(K, depth):
    vs, us = depth.nonzero()
    zs = depth[vs, us]
    #print(zs.min())
    #print(zs.max())
    xs = ((us - K[0, 2]) * zs) / float(K[0, 0])
    ys = ((vs - K[1, 2]) * zs) / float(K[1, 1])
    pts = np.array([xs, ys, zs]).T
    return pts

#for original linemod depth
def read_depth(path):
    if (path[-3:] == 'dpt'):
        with open(path) as f:
            h,w = np.fromfile(f,dtype=np.uint32,count=2)
            data = np.fromfile(f,dtype=np.uint16,count=w*h)
            depth = data.reshape((h,w))
    else:
        depth = np.asarray(Image.open(path)).copy()
    return depth


def test_epsilon(root_dataset, out_dir):
    
    for class_name in lm_cls_names:
        print ('class_name:', class_name)

        class_epsilons = []

        if not os.path.exists(out_dir + class_name):
            os.makedirs(out_dir + class_name)
        

        # Root paths for dataset and file list
        rootPath = root_dataset + "LINEMOD_ORIG/"+class_name+"/" 
        rootpvPath = root_dataset + "LINEMOD/"+class_name+"/" 
        rootRadialMapPath = root_dataset + "rkhs_estRadialMap/"+class_name+"/"
        test_list = open(root_dataset + "LINEMOD/"+class_name+"/" +"Split/val.txt","r").readlines()
        test_list = [ s.replace('\n', '') for s in test_list]
        test_list_size = len(test_list)

        # Load object pointcloud
        #pcd_load = o3d.io.read_point_cloud(root_dataset + "LINEMOD/"+class_name+"/"+class_name+".ply")
        #xyz_load = np.asarray(pcd_load.points)

        # Load KeyGNet keypoints
        keypoints=np.load(root_dataset + "rkhs_estRadialMap/KeyGNet_kpts 1.npy")
        keypoints = keypoints[0:3]
        keypoints = keypoints / 1000

        #random_files = random.sample(test_list, 5)
        for filename in tqdm(test_list, total=len(test_list), unit='image', leave=False):
            epsilons = []
            # Read in rotation and translation matrix
            RTGT = np.load(root_dataset + "LINEMOD/"+class_name+"/pose/pose"+str(int(os.path.splitext(filename)[0]))+'.npy')
            
            # transform keypoints to GT pose and mm scale
            kpGT_mm = (np.dot(keypoints, RTGT[:, :3].T) + RTGT[:, 3:].T)*1000


            keypoint_count = 0
            for keypoint in keypoints:

                centerGT_mm = kpGT_mm[keypoint_count]

                # load radial map
                radialMapPath = rootRadialMapPath + 'Out_pt'+str(keypoint_count + 1)+'_dm/'+str(filename)+'.npy'
                radMap = np.load(radialMapPath)

                # create mask for radial map
                semMask = np.where(radMap>0.8,1,0)

                # load depth map and mask it with semMask
                depthMap = read_depth(rootPath+'data/depth'+str(int(os.path.splitext(filename)[0]))+'.dpt')
                depthMap = depthMap*semMask

                # get pixel coordinates of the mask
                pixelCoords = np.where(semMask==1)

                # get radial values of the mask
                radList = radMap[pixelCoords]
                radList = radList*100

                # get 3D coordinates of the mask
                xyz_mm = rgbd_to_point_cloud(linemod_K, depthMap)

                assert xyz_mm.shape[0] == radList.shape[0], "Number of points in depth map and radial map do not match"
                assert xyz_mm.shape[0] != 0, "No points found in depth map"
                if xyz_mm.shape[0] < 50:
                    #print ('Not enough points found in depth map')
                    break

                for xyz, rad in zip(xyz_mm, radList):
                    # compute offset between GT and KeyGNet keypoints
                    offset = np.linalg.norm(xyz - centerGT_mm)

                    # compute epsilon
                    current_epsilon = abs(offset - rad)

                    epsilons.append(current_epsilon)
                    class_epsilons.append(current_epsilon)
         
                keypoint_count += 1
                if keypoint_count == 3:
                    break

            # create a histogram of the epsilon values
            #plt.hist(epsilons, bins=300)
            #plt.title('Epsilon Histogram')
            #plt.xlabel('Epsilon')
            #plt.ylabel('Frequency')
            #plt.savefig(out_dir + class_name + '/' + str(filename) + '.png')
            #plt.close() 
        
        # create a histogram of the epsilon values for the class
        # remove outliers 
        class_epsilons = np.array(class_epsilons)
        class_epsilons = class_epsilons[class_epsilons <2]
        plt.hist(class_epsilons, bins=2000)
        plt.title('Epsilon Histogram')
        plt.xlabel('Epsilon')
        plt.ylabel('Frequency')
        plt.savefig(out_dir + class_name + '/class_epsilon_with_cutoff_at_2mm.png')
        #plt.show()
        plt.close()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dataset',
                        type=str,
                        default='../../datasets/test/')
    
    out_dir = 'logs/epsilon_test/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    args = parser.parse_args()

    root_dataset = args.root_dataset

    print ('root_dataset:', args.root_dataset)
    print ('out_dir:', out_dir)
    test_epsilon(root_dataset, out_dir)
