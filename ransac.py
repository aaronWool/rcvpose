from numba import jit, prange
import numpy as np
from util.centerest import centerest

vote = np.dtype([
    ('mse', np.float64),
    ('x', np.float64),
    ('y', np.float64),
    ('z', np.float64)
])

def random_centerest(xyz, radial_list, iterations):
    vote_list = np.zeros(iterations, dtype=vote)
    for itr in prange(iterations):
        index = np.random.randint(0, len(xyz), 4)
        point_list = xyz[index]
        radius_list = radial_list[index]
        x, y, z = centerest(point_list, radius_list)
        error = 0
        for i in range (len(xyz)):
            p = xyz[i]
            r = radial_list[i]
            dist = np.sqrt((p[0]-x)*(p[0]-x) + (p[1]-y)*(p[1]-y) + (p[2]-z)*(p[2]-z))
            error += abs(dist-r)
        error /= len(xyz)
        vote_list[itr] = (error, x, y, z)

    return vote_list
    

def RANSAC_3D(xyz, radial_list):
    acc_unit = 5

    print('Number of points: ' , len(xyz))

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
       
    iterations = 100

    random_center_list = random_centerest(xyz_mm, radial_list_mm, iterations)
    
    best_vote = np.sort(random_center_list, order='mse')[0]

    xyz_mm_inliers = []
    radial_list_inliers = []
    xyz_mm_inliers = []
    radial_list_inliers = []
    for i in range(len(xyz_mm)):
        p = xyz_mm[i]
        r = radial_list_mm[i]
        dist = np.sqrt((p[0]-best_vote[1])**2 + (p[1]-best_vote[2])**2 + (p[2]-best_vote[3])**2)
        if abs(dist-r) < best_vote[0]:
            xyz_mm_inliers += [p]
            radial_list_inliers += [r]


    print('Number of removed points: ', len(xyz_mm)-len(xyz_mm_inliers))
    print('Number of remaining points: ', len(xyz_mm_inliers))

    center = centerest(xyz_mm_inliers, radial_list_inliers)

    center = np.array([center[0], center[1], center[2]])

    center = center.astype("float64")
    if(zero_boundary<0):
        center = center+zero_boundary

    center[0] = (center[0]+x_mean_mm+0.5)*acc_unit
    center[1] = (center[1]+y_mean_mm+0.5)*acc_unit
    center[2] = (center[2]+z_mean_mm+0.5)*acc_unit
    

    return center


