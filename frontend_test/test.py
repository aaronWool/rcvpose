import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import time
from ransac_3 import RANSAC_3D
import datetime
from accumulator3D import Accumulator_3D
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

#lm_cls_names = ['benchvise', 'can']
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

def estimate_6d_pose_lm(opts, iterations=2000): 
    start = 'Estimating 6D Pose on LINEMOD' 
    if opts.frontend == 'ransac' or opts.frontend == 'RANSAC':
        start += ' with RANSAC Iterations: ' + str(iterations)
    else:
        start += 'with Accumulator'
    print(start)

    debug = False
    if opts.verbose:
        debug = True

    class_accuracies = []
    class_std = []
    frontend_times = []

    totalTimeStart = time.time_ns()

    for class_name in lm_cls_names:
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

        for filename in (test_list if debug else tqdm(test_list, total=test_list_size, desc='Evaluating ' + class_name, unit='image', leave=False)):
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

                if opts.frontend == 'ransac' or opts.frontend == 'RANSAC':
                    estKP = RANSAC_3D(xyz, radList, iterations=iterations, debug=debug)
                elif opts.frontend == 'accumulator':
                    estKP = Accumulator_3D(xyz, radList)[0]

                if debug:
                    print ('Est Center: \n', estKP)


                frontend_End = time.time_ns()

                frontend_time += (frontend_End - frontend_Start)/1000000

                offset = np.linalg.norm(CenterGT_mm - estKP)
                if offset > 100000:
                    print ('\tERROR: Offset: ', offset, 'mm, Count: ', img_count, ' Keypoint: ', keypoint_count + 1, 'GT Center: ', CenterGT_mm, 'Est Center: ', estKP)
                    continue

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
                

        avg = np.mean(keypoint_offsets)
        std = np.std(keypoint_offsets)
        class_accuracies.append(avg)
        class_std.append(std)
        class_time = np.mean(classFrontendTimes)
        frontend_times.append(class_time)

        print('\tAverage' , class_name, 'Acc:\t\t', avg, 'mm')
        print('\tAverage' , class_name, 'Std:\t\t', std, 'mm')
        print('\tAverage', class_name, 'FPS:\t\t', (1 / class_time) * 1000, '\n')


    totalTimeEnd = time.time_ns()
    totalTime = (totalTimeEnd - totalTimeStart)/1000000

    avg_total_frontend_time = np.mean(frontend_times)

    mean = np.mean(class_accuracies)
    std = np.mean(class_std)
    fps = (1 / avg_total_frontend_time) * 1000

    print('Total Time: ', str(datetime.timedelta(milliseconds=totalTime)))
    print ('Average Acc: ', mean, 'mm')
    print ('Average Std: ', std, 'mm')
    print ('Average Frontend Time: ', avg_total_frontend_time, 'ms')
    print ('Average FPS', fps)

    return mean, std, fps


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dataset',
                    type=str,
                    default='../../datasets/test/')

    parser.add_argument('--frontend',
                    type=str,
                    default='RANSAC')
    # accumulator, ransac, RANSAC
    parser.add_argument('--verbose',
                        type=bool,
                    default=False)
    
    
    parser.add_argument('--out_plot',
                        type=str,
                    default='RANSAC_3'
                    )
    
    opts = parser.parse_args()
    print ('Root Dataset: ' + opts.root_dataset)
    if opts.verbose:
        print ('Verbose: ', opts.verbose)
    print ('Frontend: ' + opts.frontend)

    iterations = []
    means = []
    stds = []
    fpss = []
    
    iteration_list = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

    if opts.frontend == 'ransac' or opts.frontend == 'RANSAC':
        for itr in iteration_list:
            iterations.append(itr)
            mean, std, fps = estimate_6d_pose_lm(opts, itr) 
            means.append(mean)
            stds.append(std)
            fpss.append(fps)
            print('='*50)
            print('\n')
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))

            ax1.plot(iterations, means, label='Mean')
            ax1.set_ylabel('Mean')
            ax1.legend()

            ax2.plot(iterations, stds, label='Std')
            ax2.set_ylabel('Std')
            ax2.legend()

            ax3.plot(iterations, fpss, label='FPS')
            ax3.set_ylabel('FPS')
            ax3.set_xlabel('Iterations')
            ax3.legend()

            plt.savefig(opts.out_plot)
    else:
        estimate_6d_pose_lm(opts)
        



