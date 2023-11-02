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
import datetime
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
    
    RANSAC_itr = opts.ransac_iterations

    class_accuracies = []
    class_std = []
    frontend_times = []
    
    totalTimeStart = time.time_ns()

    for class_name in lm_cls_names:
        print("Evaluation on ", class_name)
        rootPath = opts.root_dataset + "LINEMOD_ORIG/"+class_name+"/" 
        rootpvPath = opts.root_dataset + "LINEMOD/"+class_name+"/" 
        rootRadialMapPath = opts.root_dataset + "rkhs_estRadialMap/"+class_name+"/"
        
        test_list = open(opts.root_dataset + "LINEMOD/"+class_name+"/" +"Split/val.txt","r").readlines()
        test_list = [ s.replace('\n', '') for s in test_list]

        test_list_size = len(test_list)
        
        classFrontendTimes = []

        keypoint_offsets = []
        
        #keypoints = np.load(opts.root_dataset + "LINEMOD/"+class_name+"/Outside9.npy")
        keypoints=np.load(opts.root_dataset + "rkhs_estRadialMap/KeyGNet_kpts 1.npy")
        keypoints = keypoints[0:3]

        keypoints = keypoints / 1000

        if debug:
            print('keypoints: \n', keypoints)

        dataPath = rootpvPath + 'JPEGImages/'

        img_count = 0

        for filename in test_list:
            if debug:
                print("\nEvaluating ", filename)

            RTGT = np.load(opts.root_dataset + "LINEMOD/"+class_name+"/pose/pose"+str(int(os.path.splitext(filename)[0]))+'.npy')
            
            kpGT_mm = (np.dot(keypoints, RTGT[:, :3].T) + RTGT[:, 3:].T)*1000

            img_kpt_offsets = []
            
            frontend_time = 0

            imgStart = time.time_ns()

            keypoint_count = 0
            for keypoint in keypoints:
           
                CenterGT_mm = kpGT_mm[keypoint_count]

                if debug:
                    print ('Keypoint: ', keypoint_count + 1)
                    print ('GT Center: \n', CenterGT_mm)

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
                    estKP = RANSAC_3D(xyz, radList, iterations=RANSAC_itr)
                elif opts.frontend == 'accumulator':
                    estKP = Accumulator_3D(xyz, radList)[0]

                if debug:
                    print ('Est Center: \n', estKP)


                frontend_End = time.time_ns()

                frontend_time += (frontend_End - frontend_Start)/1000000

                offset = np.linalg.norm(CenterGT_mm - estKP)

                if debug:
                    print ('Offset: ', offset)

                keypoint_offsets.append(offset)
                img_kpt_offsets.append(offset)
               
                keypoint_count+=1
                
                if (keypoint_count==3):
                    break

            imgEnd = time.time_ns()
            img_count += 1
            classFrontendTimes.append(frontend_time)
            img_acc = np.mean(img_kpt_offsets)
            img_std = np.std(img_kpt_offsets)
            total_acc = np.mean(keypoint_offsets)
            total_std = np.std(keypoint_offsets)

            imgTime = (imgEnd - imgStart)/1000000

            if debug:
                print('Frontend Time: ', frontend_time)
                print('Image Time: ', imgTime)
                print('Image Acc: ', img_acc)
                print('Image Std: ', img_std)
                print('Total Acc: ', total_acc)
                print('Total Std: ', total_std)
                wait = input("PRESS ENTER TO CONTINUE.")
            else:
                avg_frontend_time = np.mean(classFrontendTimes)
                print('\r', img_count, '/', test_list_size,': Current', class_name, 'avg acc:', total_acc, 'mm, avg std:', total_std, ', FPS:', (1 / avg_frontend_time) * 1000, '\t\t', end='', flush=True)

                
        avg = np.mean(keypoint_offsets)
        std = np.std(keypoint_offsets)
        class_accuracies.append(avg)
        class_std.append(std)
        class_time = np.mean(classFrontendTimes)
        frontend_times.append(class_time)

        print('\nAverage' , class_name, ' Accuracy: ', avg, 'mm')
        print('Average' , class_name, ' Std: ', std, 'mm')
        print('Average', class_name, 'FPS: ', (1 / class_time) * 1000, '\n')


    totalTimeEnd = time.time_ns()
    totalTime = (totalTimeEnd - totalTimeStart)/1000000

    avg_total_frontend_time = np.mean(frontend_times)

    print('Total Time: ', str(datetime.timedelta(milliseconds=totalTime)))
    print ('Average Accuracy: ', np.mean(class_accuracies), 'mm')
    print ('Average Std: ', np.mean(class_std), 'mm')
    print ('Average Frontend Time: ', avg_total_frontend_time, 'ms')
    print ('Average FPS', (1 / avg_total_frontend_time) * 1000, '\n')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dataset',
                    type=str,
                    default='C:/Users/User/.cw/work/datasets/test/')

    parser.add_argument('--frontend',
                    type=str,
                    default='ransac')
    # accumulator, ransac
    parser.add_argument('--verbose',
                        type=bool,
                    default=False)
    
    parser.add_argument('--ransac_iterations', '-ri',
                    type=int,
                    default=500)
    
    parser.add_argument('--out_file',
                        type=str,
                    default='output.txt'
                    )
    
    opts = parser.parse_args()
    print ('Root Dataset: ' + opts.root_dataset)
    if opts.verbose:
        print ('Verbose: ', opts.verbose)
    print ('Frontend: ' + opts.frontend)
    if opts.frontend == 'ransac':
        print ('RANSAC Iterations: ', opts.ransac_iterations)


    estimate_6d_pose_lm(opts) 

