import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numba import jit, njit, cuda
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
from test import rgbd_to_point_cloud, read_depth, depthList, lm_cls_names, lmo_cls_names, lm_syms, linemod_K

linemod_K = np.array([[572.4114, 0., 325.2611],
                      [0., 573.57043, 242.04899],
                      [0., 0., 1.]])



def estimate_6d_pose_lm(opts, RANSAC_itr): 
    debug = opts.verbose
    class_name = opts.class_name
    print('Estimating 6D Pose on LINEMOD for', class_name ,'object with RANSAC iterations:', RANSAC_itr)

    accuracies = []
    img_times = []

    rootPath = opts.root_dataset + "LINEMOD_ORIG/"+class_name+"/" 
    rootpvPath = opts.root_dataset + "LINEMOD/"+class_name+"/" 
    rootRadialMapPath = opts.root_dataset + "rkhs_estRadialMap/"+class_name+"/"
        
    rootPath = opts.root_dataset + "LINEMOD_ORIG/"+class_name+"/" 

    keypoints=np.load(opts.root_dataset + "rkhs_estRadialMap/KeyGNet_kpts 1.npy")
    keypoints = keypoints[0:3]

    keypoints = keypoints / 1000

    test_list = open(rootpvPath + "Split/val.txt", "r").readlines()
    test_list = [s.strip() for s in test_list]

    size = len(test_list)
    count = 0
    for filename in test_list:
        if debug:
                print("\nEvaluating ", filename)

        RTGT = np.load(opts.root_dataset + "LINEMOD/"+class_name+"/pose/pose"+str(int(os.path.splitext(filename)[0]))+'.npy')
            
        kpGT_mm = (np.dot(keypoints, RTGT[:, :3].T) + RTGT[:, 3:].T)*1000

        img_kpt_offsets = []
            
        frontend_time = 0

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
            
            estKP = np.array([0,0,0])

            frontend_start = time.time_ns()

            estKP = RANSAC_3D(xyz, radList, iterations=RANSAC_itr)

            frontend_end = time.time_ns()

            frontend_time += (frontend_end - frontend_start)/1000000

            if debug:
                print ('Est Center: \n', estKP)


            offset = np.linalg.norm(CenterGT_mm - estKP)

            if debug:
                print ('Offset: ', offset)

            img_kpt_offsets.append(offset)
               
            keypoint_count+=1
                
            if (keypoint_count==3):
                break
        

        img_end = time.time_ns()
        count += 1

        img_time = (img_end - frontend_start)/1000000

        fps = (1 / frontend_time) * 1000 if frontend_time > 0 else float('inf')

        img_times.append((frontend_time))
        current_acc = np.mean(img_kpt_offsets)
        accuracies.append(current_acc)

        print('\r', count, '/', size,': Iterations:', RANSAC_itr, ', avg acc:', np.mean(accuracies), 'mm, avg std:', np.std(accuracies), ', FPS:',fps, '\t\t', end='', flush=True)

    # Calculate overall metrics for 'ape'
    print('\r                                                                                                                                                                     ', end='', flush=True)
    avg_accuracy = np.mean(accuracies)
    std_dev = np.std(accuracies)
    avg_frontend_time = np.mean(img_times)
    total_fps = (1 / avg_frontend_time) * 1000 if avg_frontend_time > 0 else float('inf')

    print(f'\rAverage Accuracy for {opts.class_name}: {avg_accuracy:.5f} mm')
    print(f'Standard Deviation for {opts.class_name}: {std_dev:.5f} mm')
    print(f'FPS for {opts.class_name}: {total_fps:.5f}\n\n')

    return avg_accuracy, std_dev, fps

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dataset', type=str, default='C:/Users/User/.cw/work/datasets/test/')
    parser.add_argument('--verbose',
                        type=bool,
                    default=False)
    parser.add_argument('--class_name', type=str, default='ape')
    parser.add_argument('--ransac_iterations_list', nargs='+', type=int, 
                        default=[500 ,750, 1000, 1250, 1500, 1750, 2000, 2250, 2500])

    opts = parser.parse_args()

    print ('Root Dataset: ' + opts.root_dataset)
    if opts.verbose:
        print ('Verbose: ', opts.verbose)
    print ('Class Name: ' + opts.class_name)
    print ('RANSAC Iterations List: ', opts.ransac_iterations_list)

    
    all_accuracies = []
    all_std_devs = []
    all_fps = []

    for RANSAC_itr in opts.ransac_iterations_list:
        avg_accuracy, std_dev, fps = estimate_6d_pose_lm(opts, RANSAC_itr)
        all_accuracies.append(avg_accuracy)
        all_std_devs.append(std_dev)
        all_fps.append(fps)

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(opts.ransac_iterations_list, all_accuracies, marker='o')
    plt.title('RANSAC Iterations vs. Metrics')
    plt.ylabel('Mean Accuracy (mm)')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(opts.ransac_iterations_list, all_std_devs, marker='o', color='orange')
    plt.ylabel('Standard Deviation (mm)')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(opts.ransac_iterations_list, all_fps, marker='o', color='green')
    plt.ylabel('FPS')
    plt.xlabel('RANSAC Iterations')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
        
