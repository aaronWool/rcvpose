import numpy as np
from PIL import Image
import matplotlib.pyplot  as plt
import os
import time
<<<<<<< HEAD
from ransac_vanilla import RANSAC_vanilla
from ransac import RANSAC_3D
from ransac_to_accumulator import RANSAC_Accumulator
=======
from ransac_4 import ransac_4
>>>>>>> 65ecd57d254d5628ff0700520e808b9165a276e9
import datetime
from accumulator3D import Accumulator_3D
from tqdm import tqdm
import open3d as o3d

import warnings
warnings.filterwarnings("ignore")

lm_cls_names = ['ape', 'benchvise', 'cam', 'can', 'cat', 'duck', 'driller', 'eggbox', 'glue', 'holepuncher','iron','lamp','phone']

#lm_cls_names = ['benchvise', 'can']
<<<<<<< HEAD
#lm_cls_names = ['phone']
=======
#lm_cls_names = ['cam', 'can', 'cat', 'duck', 'driller', 'eggbox', 'glue', 'holepuncher','iron','lamp','phone']
>>>>>>> 65ecd57d254d5628ff0700520e808b9165a276e9
#lm_cls_names = ['ape']

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

def estimate_6d_pose_lm(opts, iterations=2000, epsilon=5, out_dir=None): 
    start = 'Estimating 6D Pose on LINEMOD' 
    if opts.frontend == 'ransac' or opts.frontend == 'RANSAC':
        start += ' with RANSAC Iterations: ' + str(iterations)
    elif opts.frontend == 'ransac_to_accumulator':
        start += ' with RANSAC to Accumulator'
    else:
        start += ' with Accumulator'
    print(start)


    debug = False
    if opts.verbose:
        debug = True


    class_accuracies = []
    class_std = []
    avg_point_count = []
    frontend_times = []
    failure_count = 0

    totalTimeStart = time.time_ns()

    for class_name in lm_cls_names:
        rootPath = opts.root_dataset + "LINEMOD_ORIG/"+class_name+"/" 
        rootpvPath = opts.root_dataset + "LINEMOD/"+class_name+"/" 
        rootRadialMapPath = opts.root_dataset + "rkhs_estRadialMap/"+class_name+"/"
        
        test_list = open(opts.root_dataset + "LINEMOD/"+class_name+"/" +"Split/val.txt","r").readlines()
        test_list = [ s.replace('\n', '') for s in test_list]

        test_list_size = len(test_list)
        
        too_few_points = 0
        total_inaccurate = 0

        keypoint_offsets = []

        pcd_load = o3d.io.read_point_cloud(opts.root_dataset + "LINEMOD/"+class_name+"/"+class_name+".ply")
        xyz_load = np.asarray(pcd_load.points)
        
        #keypoints_orig = np.load(opts.root_dataset + "LINEMOD/"+class_name+"/Outside9.npy")
        keypoints=np.load(opts.root_dataset + "rkhs_estRadialMap/KeyGNet_kpts 1.npy")
        keypoints = keypoints[0:3]

        keypoints = keypoints / 1000

        if debug:
            print('keypoints: \n', keypoints)

        dataPath = rootpvPath + 'JPEGImages/'
        
        max_radii_dm = np.zeros(3)
        for i in range(3):
            dsitances = ((xyz_load[:,0]-keypoints[i,0])**2+(xyz_load[:,1]-keypoints[i,1])**2+(xyz_load[:,2]-keypoints[i,2])**2)**0.5 
            max_radii_dm[i] = dsitances.max()*10
        if debug:
            print ('max_radii_dm: ', max_radii_dm)

    
        for filename in (test_list if debug else tqdm(test_list, total=test_list_size, desc='Evaluating ' + class_name, unit='image', leave=False)):
            if debug:
                print("\nEvaluating ", filename)
            
            #if not filename == '000090':
            #    continue

            RTGT = np.load(opts.root_dataset + "LINEMOD/"+class_name+"/pose/pose"+str(int(os.path.splitext(filename)[0]))+'.npy')
            
            kpGT_mm = (np.dot(keypoints, RTGT[:, :3].T) + RTGT[:, 3:].T)*1000

            times_kpt = []

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

                num_zero1 = np.count_nonzero(semMask==0)

                # if using accumulator space, remove all radial values outside of max radius
                if opts.frontend == 'accumulator':
                    newSemMask = np.where(radMap<=max_radii_dm[keypoint_count], semMask,0)
                    if not newSemMask.sum() == 0:
                        semMask = newSemMask
                        radMap = np.where(radMap<=max_radii_dm[keypoint_count], radMap,0)

                num_zero2 = np.count_nonzero(semMask==0)

                depthMap = read_depth(rootPath+'data/depth'+str(int(os.path.splitext(filename)[0]))+'.dpt')

                depthMap = depthMap*semMask

                pixelCoords = np.where(semMask==1)

                radList = radMap[pixelCoords]

                xyz_mm = rgbd_to_point_cloud(linemod_K, depthMap)

                avg_point_count = np.append(avg_point_count, xyz_mm.shape[0])

                if debug and opts.frontend == 'accumulator':
                    print ('Number of removed radial val outside max radius: ', num_zero2 - num_zero1)
                    print ('Number of points in depth map: ', xyz_mm.shape[0])

                if xyz_mm.shape[0] == 0 and not debug:
                    print ('Number of removed radial val outside max radius: ', num_zero2 - num_zero1)
                    print ('Number of points in depth map: ', xyz_mm.shape[0])

                xyz = xyz_mm / 1000

                assert xyz.shape[0] == radList.shape[0], "Number of points in depth map and radial map do not match"
                assert xyz.shape[0] != 0, "No points found in depth map"

                if xyz.shape[0] < 200:
                    too_few_points += 1
                    #print ()
                    #print ('Not enough points found in depth map')
                    #print ('Filename: ', filename)
                    #print ('Keypoint: ', keypoint_count + 1)
                    #print ('Number of points in depth map: ', xyz_mm.shape[0])
                    keypoint_count+=1
                    if keypoint_count == 3:
                        break
                    continue
 
                estKP = np.array([0,0,0])

                if opts.frontend == 'ransac' or opts.frontend == 'RANSAC':
                    frontend_Start = time.time_ns()
<<<<<<< HEAD
                    estKP = RANSAC_vanilla(xyz, radList, epsilon=epsilon, iterations=iterations, debug=debug)
=======
                    estKP = ransac_4(xyz, radList, epsilon=epsilon, iterations=iterations, debug=debug)
>>>>>>> 65ecd57d254d5628ff0700520e808b9165a276e9
                    frontend_End = time.time_ns()

                elif opts.frontend == 'ransac_to_accumulator':
                    frontend_Start = time.time_ns()
                    estKP = RANSAC_Accumulator(xyz, radList, epsilon=epsilon, iterations=25, debug=debug)
                    frontend_End = time.time_ns()

                elif opts.frontend == 'accumulator':
                    frontend_Start = time.time_ns()
                    estKP = Accumulator_3D(xyz, radList)[0]
                    frontend_End = time.time_ns()
                    
                    if debug and (frontend_End - frontend_Start)/1000000 > 100:
                        print ('Filename: ', filename)
                        print ('Keypoint: ', keypoint_count + 1)
                        print ('Number of removed radial val outside max radius: ', num_zero2 - num_zero1)
                        print ('Number of points in depth map: ', xyz_mm.shape[0])
                        print ('Frontend Time: ', (frontend_End - frontend_Start)/1000000, 'ms')
                
             

                if debug:
                    print ('Est Center: \n', estKP)
                    print ('Frontend Time: ', (frontend_End - frontend_Start)/1000000, 'ms')
                

                frontend_time = (frontend_End - frontend_Start)/1000000

                times_kpt.append(frontend_time)

                offset = np.linalg.norm(CenterGT_mm - estKP)

                if debug:
                    print ('Offset: ', offset, 'mm')
                    #if offset > 10:
                    #    wait = input("PRESS ENTER TO CONTINUE.")
                if offset > 10000:
                    print ()
                    print ('Failed to estimate keypoint')
                    print ('Filename: ', filename)
                    print ('\nOffset: ', offset, 'mm')
                    print ('GT Center: \n', CenterGT_mm)
                    print ('Est Center: \n', estKP)
                    print ('Keypoint: ', keypoint_count + 1)
                    print ('Number of points in depth map: ', xyz_mm.shape[0])
                    total_inaccurate += 1
                    #wait = input("PRESS ENTER TO CONTINUE.")
                    continue

                keypoint_offsets.append(offset)
               
                keypoint_count+=1
                
                if (keypoint_count==3):
                    break

      
        avg = np.mean(keypoint_offsets)
        std = np.std(keypoint_offsets)
        class_accuracies.append(avg)
        class_std.append(std)
        class_time = np.mean(times_kpt)
        frontend_times.append(class_time)
        avg_point_count = np.mean(avg_point_count)
        failure_count += total_inaccurate

        print('\tAverage' , class_name, 'Acc:\t\t', avg, 'mm')
        print('\tAverage' , class_name, 'Std:\t\t', std, 'mm')
        print('\tAverage', class_name, 'FPS:\t\t', (1 / class_time) * 1000)
        print('\tAverage Number of Points: \t', avg_point_count)
        print ('\tNumber of Images with too few points: ', too_few_points)
        print ('\tNumber of Inaccurate Estimations: ', total_inaccurate)
        print()
        
        if opts.frontend == 'accumulator' or opts.frontend == 'ransac_to_accumulator':
            with open(out_dir + 'results.txt', 'a') as file:
                file.write(f"{class_name} Mean: {avg}, Std: {std}, FPS: {(1 / class_time) * 1000}\n")

                


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
    print ('Number of Inaccurate Estimations: ', failure_count)
    print()

    return mean, std, fps


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dataset',
                    type=str,
                    default='../../datasets/test/')
    # 'D:/' '../../datasets/test/'
    parser.add_argument('--frontend',
                    type=str,
                    default='ransac')   
    # accumulator, ransac, RANSAC, ransac_to_accumulator
    parser.add_argument('--verbose',
                    type=bool,
                    default=False)
    parser.add_argument('--out_dir',
                    type=str,
                    default='vanilla_ransac_25_iterations_finetuned_epsilons_200_min_points')
    

    opts = parser.parse_args()

    out_dir = 'logs/' + opts.frontend + '/' + opts.out_dir

    num_logs = len(os.listdir('logs/' + opts.frontend + '/')) + 1

    out_dir += str(num_logs) + '/'
   
    print ('Root Dataset: ' + opts.root_dataset)
    print ('Out Dir: ' + out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if opts.verbose:
        print ('Verbose: ', opts.verbose)
    print ('Frontend: ' + opts.frontend)
    print()

    if opts.frontend == 'ransac' or opts.frontend == 'RANSAC':
        eps_list = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        curr_eps = []
        means, stds, fpss = [], [], []
        print ('Epsilon List: ', eps_list)
        for eps in eps_list:
            mean, std, fps = estimate_6d_pose_lm(opts, iterations=200, epsilon=eps, out_dir=out_dir)
            print(f"Epsilon: {eps} Mean: {mean}, Std: {std}, FPS: {fps}")
            print ('='*50)
            means.append(mean)
            stds.append(std)
            fpss.append(fps)
            curr_eps.append(eps)
            with open(out_dir + 'results.txt', 'a') as file:
                file.write(f"Epsilon: {eps} Mean: {mean}, Std: {std}, FPS: {fps}\n")
            # plot the mean on the left hand side and std on the right hand side for each epsilon
            plt.figure(figsize=(15,5))
            plt.subplot(1,2,1)
            plt.plot(curr_eps, means, 'r')
            plt.title('Mean vs Epsilon')
            plt.xlabel('Epsilon')
            plt.ylabel('Mean')
            plt.subplot(1,2,2)
            plt.plot(curr_eps, stds, 'b')
            plt.title('Std vs Epsilon')
            plt.xlabel('Epsilon')
            plt.ylabel('Std')
            plt.savefig(out_dir + 'mean_std_vs_epsilon.png')
            plt.close()        
    else: 
        mean, std, fps = estimate_6d_pose_lm(opts, iterations=25, epsilon=0.7, out_dir=out_dir)
        with open(out_dir + 'results.txt', 'a') as file:
            file.write(f"Mean: {mean}, Std: {std}, FPS: {fps}\n")
    print('='*50)



        



