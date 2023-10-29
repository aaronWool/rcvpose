import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numba import jit,njit,cuda
import os
import open3d as o3d
import time
from ransac import RANSAC_3D
from numba import prange
import math
import h5py
from sklearn import metrics
import scipy
from accumulator3D import Accumulator_3D


lm_cls_names = ['ape', 'benchvise', 'cam', 'can', 'cat', 'duck', 'driller', 'eggbox', 'glue', 'holepuncher','iron','lamp','phone']
lmo_cls_names = ['ape', 'can', 'cat', 'duck', 'driller',  'eggbox', 'glue', 'holepuncher']

lm_syms = ['eggbox', 'glue']



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


depthList=[]

def estimate_6d_pose_lm(opts): 
    print('Estimating 6D Pose on LINEMOD')  
    debug = False
    if opts.verbose:
        debug = True
    
    imageTimes = []
    frontendTimes = []

    accuracies = []
    
    totalTimeStart = time.time_ns()

    for class_name in lm_cls_names:
        print("Evaluation on ", class_name)
        rootPath = opts.root_dataset + "LINEMOD_ORIG/"+class_name+"/" 
        rootpvPath = opts.root_dataset + "LINEMOD/"+class_name+"/" 
        rootRadialMapPath = opts.root_dataset + "rkhs_estRadialMap/"+class_name+"/"
        
        test_list = open(opts.root_dataset + "LINEMOD/"+class_name+"/" +"Split/val.txt","r").readlines()
        test_list = [ s.replace('\n', '') for s in test_list]
        
        classFrontendTimes = []

        image_accuracies = []
        
        keypoints=np.load(opts.root_dataset + "rkhs_estRadialMap/KeyGNet_kpts 1.npy")
        #keypoints = np.load(opts.root_dataset + "LINEMOD/"+class_name+"/Outside9.npy")

        dataPath = rootpvPath + 'JPEGImages/'

        for filename in test_list:
            print("\nEvaluating ", filename)

            RTGT = np.load(opts.root_dataset + "LINEMOD/"+class_name+"/pose/pose"+str(int(os.path.splitext(filename)[0]))+'.npy')

            
            kpGT_mm = (np.dot(keypoints, RTGT[:, :3].T) + RTGT[:, 3:].T)*1000
            
            frontend_avg_time = 0

            imgStart = time.time_ns()

            estAcc = np.zeros((3,1))
            estKPs = np.zeros((3,3))

            keypoint_count = 0
            for keypoint in keypoints:
           
                CenterGT_mm = kpGT_mm[keypoint_count]

                radialMapPath = rootRadialMapPath + 'Out_pt'+str(keypoint_count + 1)+'_dm/'+str(filename)+'.npy'

                #radialMapPath = rootpvPath + 'Out_pt'+str(keypoint_count + 1)+'_dm/'+str(filename)+'.npy'

                radMap = np.load(radialMapPath)

                semMask = np.where(radMap>0.8,1,0)

                depthMap = read_depth(rootPath+'data/depth'+str(int(os.path.splitext(filename)[0]))+'.dpt')

                depthMap = depthMap*semMask

                pixelCoords = np.where(semMask==1)

                radList = radMap[pixelCoords]

                xyz_mm = rgbd_to_point_cloud(linemod_K, depthMap)

                xyz = xyz_mm / 1000

                assert xyz.shape[0] == radList.shape[0]

                frontend_Start = time.time_ns()
                
                estKP = np.array([0,0,0])

                if opts.frontend == 'ransac':
                    estKP = RANSAC_3D(xyz, radList)
                elif opts.frontend == 'accumulator':
                    estKP = Accumulator_3D(xyz, radList)

                frontend_End = time.time_ns()

                frontend_avg_time += (frontend_End - frontend_Start)/1000000

                classFrontendTimes.append((frontend_End - frontend_Start)/1000000)

                offset = np.linalg.norm(CenterGT_mm - estKP)

                estAcc[keypoint_count] = offset
                estKPs[keypoint_count] = estKP[0]
                
                keypoint_count+=1
                if (keypoint_count==3):
                    break

            imgEnd = time.time_ns()
            imgTime = (imgEnd - imgStart)/1000000

            imageTimes.append(imgTime)
            image_accuracies.append(estAcc)

            image_accuracy = estAcc.mean(axis=0)

            if debug:
                print('GT Keypoints: \n', kpGT_mm)
                print('Estimated Keypoints: \n', estKPs)
                print('Accuracies: \n', estAcc)
                print('Image Accuracy: ', image_accuracy)
                print('Image Time: ', imgTime)
                print('Total Frontend Time: ', frontend_avg_time)
                print('Average Frontend Time: ', frontend_avg_time / 3)
                

        class_time = classFrontendTimes.mean(axis=0)
        frontendTimes.append(class_time)
        avgClassAccuracy = image_accuracies.mean(axis=0)
        print('Average' , class_name, ' Accuracy: ', avgClassAccuracy)
        accuracies.append(avgClassAccuracy)
    
    totalTimeEnd = time.time_ns()
    totalTime = (totalTimeEnd - totalTimeStart)/1000000
    print('Total Time: \n\t', totalTime)
    print('Image Times: \n', imageTimes)
    print('Frontend Times: \n', frontendTimes)
    print('Accuracies: \n', accuracies)

            
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dataset',
                    type=str,
                    default='C:/Users/User/.cw/work/datasets/test/')

    parser.add_argument('--frontend',
                    type=str,
                    default='accumulator')
    
    parser.add_argument('--verbose',
                        type=bool,
                    default=True)

    opts = parser.parse_args()
    print ('Root Dataset: ' + opts.root_dataset)
    print ('Frontend: ' + opts.frontend)
    estimate_6d_pose_lm(opts) 

