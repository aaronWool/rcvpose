import numpy as np
from PIL import Image
import matplotlib.pyplot  as plt
import os
import time
from ransac_4 import RANSAC_3D
import datetime
from accumulator3D import Accumulator_3D
from tqdm import tqdm
import open3d as o3d
from numba import jit, prange, njit
import warnings
warnings.filterwarnings("ignore")

lm_cls_names = ['ape', 'benchvise', 'cam', 'can', 'cat', 'duck', 'driller', 'eggbox', 'glue', 'holepuncher','iron','lamp','phone']

#lm_cls_names = ['benchvise', 'can']
#lm_cls_names = ['cam', 'can', 'cat', 'duck', 'driller', 'eggbox', 'glue', 'holepuncher','iron','lamp','phone']
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

from numba import jit, prange, njit
import numpy as np
import random


@jit(nopython=True)
def centerest(point_list, radius_list):
    assert len(point_list) == len(radius_list), 'different number of points and radii'
    assert len(point_list) >= 4, 'less than 4 points'

    A = np.zeros((len(point_list), 5))
    b = np.zeros((len(point_list), 5))
    for i in prange(len(point_list)):
        p = point_list[i]
        r = radius_list[i]
        x = p[0]
        y = p[1]
        z = p[2]
        A[i] = [-2*x, -2*y, -2*z, 1, x*x+y*y+z*z-r*r]
        b[i] = [0, 0, 0, 0, 0]

    U, S, Vh = np.linalg.svd(A)
    X = Vh[-1]
    X /= X[-1]

    return X[0], X[1], X[2]


# Random Consensus looks for a good first guess, uses epsilon to determine inliers
@jit(nopython=True, parallel=True)
def random_centerest(xyz, radial_list, iterations, epsilon, consensus_count = 400, debug=False):
    
    xyz_len = len(xyz)
    n = consensus_count

    votes = np.zeros((iterations, 4))

    for itr in prange(iterations):
        index = np.random.randint(0, xyz_len, 4)

        point_list = xyz[index]

        radius_list = radial_list[index]
        
        x, y, z = centerest(point_list, radius_list)

        error = 0

        for i in prange(n):
            idx2 = np.random.randint(0, xyz_len)
            p = xyz[idx2]
            r = radial_list[idx2]
            dist = ((p[0]-x)**2 + (p[1]-y)**2 + (p[2]-z)**2)**0.5
            if abs(dist - r) <= epsilon:
                error += 1

        votes[itr, 0] = error
        votes[itr, 1] = x
        votes[itr, 2] = y
        votes[itr, 3] = z

    
    sorted_votes = sorted(votes, key=lambda x: x[0], reverse=True)

    best_vote = sorted_votes[0]
    
    return best_vote



@njit(parallel=True)
def calc_inliers(xyz, radial_list, center, epsilon):
    inlier_count = 0
    for i in prange(len(xyz)):
        p = xyz[i]
        r = radial_list[i]
        dist = ((p[0]-center[0])**2 + (p[1]-center[1])**2 + (p[2]-center[2])**2)**0.5
        if abs(dist - r) <= epsilon:
            inlier_count += 1

    return inlier_count




def estimate_and_accumulate(xyz, radial_list, iterations=100, epsilon = 0.7, debug=False):
    acc_unit = 1

    # Shift Data 
    xyz_mm = xyz*1000/acc_unit 

    x_mean_mm = np.mean(xyz_mm[:,0])
    y_mean_mm = np.mean(xyz_mm[:,1])
    z_mean_mm = np.mean(xyz_mm[:,2])

    xyz_mm[:,0] -= x_mean_mm
    xyz_mm[:,1] -= y_mean_mm
    xyz_mm[:,2] -= z_mean_mm

    radial_list_mm = radial_list*100/acc_unit  

    xyz_mm_min = xyz_mm.min()
    xyz_mm_max = xyz_mm.max()
    radius_max = radial_list_mm.max()

    zero_boundary = int(xyz_mm_min-radius_max)+1

    if(zero_boundary<0):
        xyz_mm -= zero_boundary

    best_vote = random_centerest(xyz_mm, radial_list_mm, iterations, epsilon, debug=debug)

    center = np.array([best_vote[1], best_vote[2], best_vote[3]])

    inlier_count = calc_inliers(xyz_mm, radial_list_mm, center, epsilon)

    return inlier_count 




 


def test_ransac_inliers(opts, epsilon=1.0, iterations=1000): 
    print ('Evaluating RANSAC Inliers')

    debug = False
    if opts.verbose:
        debug = True


    total_pixel_count = []
    total_percent_inliers = []
    total_inlier_count = []

    for class_name in lm_cls_names:
        class_pixel_count = []
        class_percent_inliers = []
        class_inlier_count = []
        rootPath = opts.root_dataset + "LINEMOD_ORIG/"+class_name+"/" 
        rootpvPath = opts.root_dataset + "LINEMOD/"+class_name+"/" 
        rootRadialMapPath = opts.root_dataset + "rkhs_estRadialMap/"+class_name+"/"
        
        test_list = open(opts.root_dataset + "LINEMOD/"+class_name+"/" +"Split/val.txt","r").readlines()
        test_list = [ s.replace('\n', '') for s in test_list]

        test_list_size = len(test_list)

        keypoints=np.load(opts.root_dataset + "rkhs_estRadialMap/KeyGNet_kpts 1.npy")
        keypoints = keypoints[0:3]

        keypoints = keypoints / 1000

        if debug:
            print('keypoints: \n', keypoints)

        dataPath = rootpvPath + 'JPEGImages/'

    
        for filename in (test_list if debug else tqdm(test_list, total=test_list_size, desc='Evaluating ' + class_name, unit='image', leave=False)):
            if debug:
                print("\nEvaluating ", filename)
            

            RTGT = np.load(opts.root_dataset + "LINEMOD/"+class_name+"/pose/pose"+str(int(os.path.splitext(filename)[0]))+'.npy')
            
            kpGT_mm = (np.dot(keypoints, RTGT[:, :3].T) + RTGT[:, 3:].T)*1000

            keypoint_count = 0
            for keypoint in keypoints:
           
                CenterGT_mm = kpGT_mm[keypoint_count]

                if debug:
                    print ('Keypoint: ', keypoint_count + 1)
                    print ('GT Center: \n', CenterGT_mm)

                radialMapPath = rootRadialMapPath + 'Out_pt'+str(keypoint_count + 1)+'_dm/'+str(filename)+'.npy'

                radMap = np.load(radialMapPath)

                semMask = np.where(radMap>0.8,1,0)

                depthMap = read_depth(rootPath+'data/depth'+str(int(os.path.splitext(filename)[0]))+'.dpt')

                depthMap = depthMap*semMask

                pixelCoords = np.where(semMask==1)

                radList = radMap[pixelCoords]

                xyz_mm = rgbd_to_point_cloud(linemod_K, depthMap)

                xyz = xyz_mm / 1000

                class_pixel_count.append(len(xyz_mm))

                assert xyz.shape[0] == radList.shape[0], "Number of points in depth map and radial map do not match"
                assert xyz.shape[0] != 0, "No points found in depth map"

                inliers = estimate_and_accumulate(xyz, radList, iterations=iterations, epsilon=epsilon, debug=debug)

                class_inlier_count.append(inliers)

                class_percent_inliers.append(inliers/len(xyz))
            
                keypoint_count += 1

        print ('Class: ', class_name)
        print ('Average Percent Inliers: ', np.mean(class_percent_inliers))
        print ('Average Pixel Count: ', np.mean(class_pixel_count))
        print ('Average Inlier Count: ', np.mean(class_inlier_count))
        total_pixel_count.append(np.mean(class_pixel_count))
        total_percent_inliers.append(np.mean(class_percent_inliers))
        total_inlier_count.append(np.mean(class_inlier_count))

    return np.mean(total_percent_inliers), np.mean(total_pixel_count), np.mean(total_inlier_count)
            

        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dataset',
                    type=str,
                    default='D:/')
    # 'D:/' '../../datasets/test/'

    parser.add_argument('--verbose',
                    type=bool,
                    default=False)
    parser.add_argument('--out_file',
                    type=str,
                    default='logs/inlier_tests/')
    

    opts = parser.parse_args()
    
    if not os.path.exists(opts.out_file):
        os.makedirs(opts.out_file)


    print ('Root Dataset: ' + opts.root_dataset)
    print ('Output File: ' + opts.out_file)


    epsilon = 1.0

    epsilon_decrease = 0.1

    eps = []

    percent_inliers = []

    avg_pixel_count = []

    avg_inlier_count = []

    while epsilon > 0.01:
        print ('Epsilon: ', epsilon)
        percent, pix_count, inlier_count = test_ransac_inliers(opts, epsilon=epsilon, iterations=5000)
        percent_inliers.append(percent)
        avg_pixel_count.append(pix_count)
        avg_inlier_count.append(inlier_count)
        eps.append(epsilon)

        epsilon -= epsilon_decrease

    plt.plot(eps, percent_inliers)
    plt.xlabel('Epsilon')
    plt.ylabel('Percent Inliers')
    plt.title('Epsilon vs Percent Inliers')
    plt.savefig(opts.out_file + 'Epsilon_vs_Percent_Inliers.png')
    plt.show()
    


   
    



