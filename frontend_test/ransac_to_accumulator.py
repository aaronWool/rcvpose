from numba import jit, prange, njit
import numpy as np
import random
import time

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


# Random Consensus looks for a good first guess
@jit(nopython=True, parallel=True)
def random_centerest(xyz, radial_list, iterations, epsilon, debug=False):
    
    xyz_len = len(xyz)
    n = xyz_len

    #if n > 200:
    #    n = 200

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


@jit(nopython=True,parallel=True)   
def fast_for(xyz_mm,radial_list_mm,VoteMap_3D):  
    factor = (3**0.5)/4
    for count in prange(xyz_mm.shape[0]):
        xyz = xyz_mm[count]
        radius = radial_list_mm[count]
        radius = int(np.around(radial_list_mm[count]))
        shape = VoteMap_3D.shape
        for i in prange(VoteMap_3D.shape[0]):
            for j in prange(VoteMap_3D.shape[1]):
                for k in prange(VoteMap_3D.shape[2]):
                    distance = ((i-xyz[0])**2+(j-xyz[1])**2+(k-xyz[2])**2)**0.5
                    if radius - distance < factor and radius - distance>0:
                        VoteMap_3D[i,j,k]+=1
        
    return VoteMap_3D




# Iterate through all the data points and accumulate inliers
@njit(parallel=True)
def accumulate_inliers(xyz, radial_list, iterations, best_vote, error, max_inliers=200):
    xyz_inliers = np.zeros((max_inliers, 3))  
    radial_list_inliers = np.zeros(max_inliers)
    inlier_count = 0

    indexes = np.arange(len(xyz))
    np.random.shuffle(indexes)

    for itr in prange(iterations):
        if inlier_count >= max_inliers:
            break  
        i = indexes[itr % len(xyz)]
        p = xyz[i]
        r = radial_list[i]
        dist = np.sqrt((p[0] - best_vote[1]) ** 2 + (p[1] - best_vote[2]) ** 2 + (p[2] - best_vote[3]) ** 2)
        if abs(dist - r) < error:
            xyz_inliers[inlier_count] = p
            radial_list_inliers[inlier_count] = r
            inlier_count += 1

    return xyz_inliers[:inlier_count], radial_list_inliers[:inlier_count]


def RANSAC_Accumulator(xyz, radial_list, iterations=100, epsilon = 0.7, debug=False):
    acc_unit = 5
    current_epsilon = epsilon

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
    
    eps_increment = 0

    # Random Consensus
    best_vote = random_centerest(xyz_mm, radial_list_mm, iterations, epsilon=current_epsilon,debug=debug)

    # If no consensus, increase epsilon and try again
    if best_vote[0] == 0:
        while best_vote[0] == 0:
            current_epsilon += 0.1
            eps_increment += 1
            best_vote = random_centerest(xyz_mm, radial_list_mm, iterations, epsilon=current_epsilon,debug=debug)

    if debug:
        if eps_increment > 0:
            print('\tEpsilon Times Incremented for RANSAC: ' + str(eps_increment))

    eps_increment = 0
    num_iterations = len(xyz_mm)
    current_epsilon = epsilon

    # Accumulate Inliers with all the data points
    xyz_inliers, radial_list_inliers = accumulate_inliers(xyz_mm, radial_list_mm, num_iterations , best_vote, current_epsilon)
    
    # If no inliers, increase epsilon and try again
    if xyz_inliers == []:
        while xyz_inliers == []:
            current_epsilon += 0.1
            eps_increment += 1
            xyz_inliers, radial_list_inliers = accumulate_inliers(xyz_mm, radial_list_mm, num_iterations, best_vote, current_epsilon)
     
    # Final Center incase of less than 4 inliers
    center = np.array([best_vote[1], best_vote[2], best_vote[3]])

    if debug:
        print('\tNumber of inliers: ' + str(len(xyz_inliers)))
        if eps_increment > 0:
            print('\tEpsilon Times incremented for Inliers: ' + str(eps_increment))

    # Centerest if only 4 inliers
    if len(xyz_inliers) == 4:
        center = centerest(xyz_inliers, radial_list_inliers)
        center = np.array(center)
        if debug:
            print('\tCenterest output: ', center)

    # Refine Consensus if more than 4 inliers
    if len(xyz_inliers) > 4:
        length = int(xyz_inliers.max())
        Vote_Map_3D = np.zeros((length+int(radius_max), length+int(radius_max), length+int(radius_max)))
        tic = time.perf_counter()
        Vote_Map_3D = fast_for(xyz_mm,radial_list_mm,Vote_Map_3D)
        toc = time.perf_counter()
        if debug:
            print(f"Accumulator 3D Time: {toc - tic} seconds")
        center = np.argwhere(Vote_Map_3D==Vote_Map_3D.max())
        center = center[0]
        if debug:
            print('\tRefined centerest: ', center)            
    
    # Shift Center back to original data
    center = center.astype("float64")


    if(zero_boundary<0):
        center = center+zero_boundary

    center[0] = (center[0]+x_mean_mm+0.5)*acc_unit
    center[1] = (center[1]+y_mean_mm+0.5)*acc_unit
    center[2] = (center[2]+z_mean_mm+0.5)*acc_unit

    if debug:
        print('\tFinal center after data shift: ' + str(center))
    
    return center

