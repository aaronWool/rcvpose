import numba
from numba import jit, prange, njit
import numpy as np
from util.horn import HornPoseFitting
import random


@jit(nopython=True)
def center_est(point_list, radius_list):
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


# Random Sample Consensus looks for a good first guess, uses epsilon to determine inliers
@jit(nopython=True, parallel=True)
def random_center_est(xyz, radial_list, epsilon, iterations=25):

    n = len(xyz)

    votes = np.zeros((iterations, 4))

    for itr in prange(iterations):
        index = np.random.randint(0, n, 4)

        point_list = xyz[index]

        radius_list = radial_list[index]
        
        x, y, z = center_est(point_list, radius_list)

        consensus = 0

        for i in prange(n):
            idx2 = np.random.randint(0, n)
            p = xyz[idx2]
            r = radial_list[idx2]
            dist = ((p[0]-x)**2 + (p[1]-y)**2 + (p[2]-z)**2)**0.5
            if abs(dist - r) <= epsilon:
                consensus += 1

        votes[itr, 0] = consensus
        votes[itr, 1] = x
        votes[itr, 2] = y
        votes[itr, 3] = z

    
    sorted_votes = sorted(votes, key=lambda x: x[0], reverse=True)

    best_vote = sorted_votes[0]
    
    return best_vote


# @jit(nopython=True, parallel=True)
def random_est_accumulate(xyz, radial_list, best_vote, epsilon, iterations):
    estimates = np.zeros((iterations, 3))
    count = 0

    for itr in prange(iterations):
        index = np.random.randint(0, len(xyz), 4)
        point_list = xyz[index]
        radius_list = radial_list[index]
        x, y, z = center_est(point_list, radius_list)
        dist = ((x-best_vote[1])**2 + (y-best_vote[2])**2 + (z-best_vote[3])**2)**0.5
        if abs(dist - best_vote[0]) < epsilon:
            continue

        estimates[count] = np.array([x, y, z])
        count += 1

    estimates = estimates[:count]

    return estimates





    

# Iterate through all the data points and accumulate inliers
#@njit(parallel=True)
def accumulate_inliers(xyz, radial_list, iterations, best_vote, epsilon, early_stop=None):

    if early_stop is None:
        early_stop = iterations

    xyz_inliers = np.zeros((iterations, 3))  
    radial_list_inliers = np.zeros(iterations)
    inlier_count = 0

    indexes = np.arange(len(xyz))
    np.random.shuffle(indexes)

    for itr in prange(iterations):
        if inlier_count >= early_stop:
            break
        i = indexes[itr]
        p = xyz[i]
        r = radial_list[i]
        dist = np.sqrt((p[0] - best_vote[1]) ** 2 + (p[1] - best_vote[2]) ** 2 + (p[2] - best_vote[3]) ** 2)
        if abs(dist - r) < epsilon:
            xyz_inliers[inlier_count] = p
            radial_list_inliers[inlier_count] = r
            inlier_count += 1

    return xyz_inliers[:inlier_count], radial_list_inliers[:inlier_count]


# Refine Consensus with only inliers, uses best fitting point to determine center
@jit(nopython=True, parallel=True)
def RANSAC_best_fit(xyz, radial_list, iterations):
    
    xyz_len = len(xyz)
    n = xyz_len

    votes = np.zeros((iterations, 4))

    for itr in prange(iterations):
        index = np.random.randint(0, n, 4)

        point_list = xyz[index]

        radius_list = radial_list[index]
        
        x, y, z = center_est(point_list, radius_list)

        error = 0

        for i in prange(n):
            p = xyz[i]
            r = radial_list[i]
            dist = ((p[0]-x)**2 + (p[1]-y)**2 + (p[2]-z)**2)**0.5
            error += abs(dist - r)
        
        error /= n

        votes[itr, 0] = error
        votes[itr, 1] = x
        votes[itr, 2] = y
        votes[itr, 3] = z

    
    sorted_votes = sorted(votes, key=lambda x: x[0], reverse=False)

    best_vote = sorted_votes[0]
    
    return best_vote



# Problematic function, not used
@jit(nopython=True, parallel=True)
def random_center_est_and_inliers(xyz, radial_list, iterations, epsilon, debug=False):
    n = len(xyz)

    # Votes is a 2D array with the first column being the number of inliers and the next 3 columns being the center
    # All inliers is a 3D array with the first dimension being the iteration, the second dimension being the number of inliers, and the third dimension being the 3D point
    # Inliers counts is a 1D array with the number of inliers for each iteration
    votes = np.zeros((iterations, 4), dtype=np.float64)
    all_inliers_xyz = np.zeros((iterations, n, 3), dtype=np.float64)
    all_inliers_radial = np.zeros((iterations, n), dtype=np.float64)
    inliers_counts = np.zeros(iterations, dtype=np.int64)

    # Iterate through the data and accumulate inliers
    for itr in prange(iterations):
        # Randomly select 4 points
        index = np.random.randint(0, n, 4)
        # Get the points and radii
        point_list = xyz[index]
        radius_list = radial_list[index]
        
        # Estimate the center
        x, y, z = center_est(point_list, radius_list)

        inlier_count = 0

        # Iterate through all the data points and accumulate inliers
        for i in range(n):
            p = xyz[i]
            r = radial_list[i]
            dist = np.sqrt((p[0]-x)**2 + (p[1]-y)**2 + (p[2]-z)**2)
            # If the distance between the point and the center is within epsilon of the radius, add it to the inliers
            if abs(dist - r) <= epsilon:
                all_inliers_xyz[itr, inlier_count] = p
                all_inliers_radial[itr, inlier_count] = r
                inlier_count += 1
        # Add the inlier count to the votes
        votes[itr, 0] = inlier_count
        votes[itr, 1] = x
        votes[itr, 2] = y
        votes[itr, 3] = z
        inliers_counts[itr] = inlier_count

    # Find the best vote
    best_itr = np.argmax(votes[:, 0])

    # Get the best vote and inliers
    best_vote = votes[best_itr]
    best_inliers_xyz = all_inliers_xyz[best_itr, :inliers_counts[best_itr]]
    best_inliers_radial = all_inliers_radial[best_itr, :inliers_counts[best_itr]]

    return best_vote, best_inliers_xyz, best_inliers_radial

def linear_least_squares(xyz, radial_list):
    xyz_mm = xyz*1000
    radial_list_mm = radial_list*100

    x, y, z = center_est(xyz_mm, radial_list_mm)

    return np.array([x, y, z])



def RANSAC(xyz, radial_list, iterations, epsilon):
    assert isinstance(iterations, int), 'iterations must be an integer'

    xyz_mm = xyz*1000
    radial_list_mm = radial_list*100    

    best_vote = random_center_est(xyz_mm, radial_list_mm, epsilon, iterations)

    center = np.array([best_vote[1], best_vote[2], best_vote[3]])

    center = center.astype("float64")
    
    return center


def RANSAC_geometric(initial_kpts, xyz1, xyz2, xyz3, radial_list1, radial_list2, radial_list3, iterations, epsilon, gt_kpts=None, debug=False):
    assert isinstance(iterations, int), 'iterations must be an integer'

    if debug:
        assert gt_kpts is not None, 'gt_kpts must be provided for debugging'

    horn = HornPoseFitting()

    kpt1 = np.zeros(3)
    kpt2 = np.zeros(3)
    kpt3 = np.zeros(3)

    # Convert to mm

    xyz1_mm = xyz1*1000
    xyz2_mm = xyz2*1000
    xyz3_mm = xyz3*1000

    radial_list1_mm = radial_list1*100
    radial_list2_mm = radial_list2*100
    radial_list3_mm = radial_list3*100
    

    if debug:
        print('\n')
        print ('='*50)
        print ('RANSAC: Geometric Refinement')
        # print ('\tinitial_kpts = ', initial_kpts)
        print ('\tEpsilon (mm) = ', epsilon)
        print ('\tIterations = ', iterations)
        print ('\tLength of xyz1 = ', len(xyz1))
        print ('\tLength of xyz2 = ', len(xyz2))
        print ('\tLength of xyz3 = ', len(xyz3))

    # Get the best vote and inliers for each estimated keypoint
    best_vote1 = random_center_est(xyz1_mm, radial_list1_mm, epsilon, iterations)
    best_vote2 = random_center_est(xyz2_mm, radial_list2_mm, epsilon, iterations)
    best_vote3 = random_center_est(xyz3_mm, radial_list3_mm, epsilon, iterations)

    if debug:
        print ('\tbest_vote1 = ', best_vote1)
        print ('\tbest_vote2 = ', best_vote2)
        print ('\tbest_vote3 = ', best_vote3)

    RT_temp = np.zeros((4,4))

    est_kpts = np.zeros((3,3))
    est_kpts[0] = best_vote1[1:4]
    est_kpts[1] = best_vote2[1:4]
    est_kpts[2] = best_vote3[1:4]


    horn.lmshorn(initial_kpts, est_kpts, 3, RT_temp)

    transformed_initial_keypoints = (np.dot(initial_kpts, RT_temp[:3,:3].T) + RT_temp[:3,3])

    
    new_vote1 = np.zeros(4)
    new_vote2 = np.zeros(4)
    new_vote3 = np.zeros(4)

    new_vote1[1:4] = transformed_initial_keypoints[0]
    new_vote2[1:4] = transformed_initial_keypoints[1]
    new_vote3[1:4] = transformed_initial_keypoints[2]

    
    if debug:
        print ('\tvote with geo constraint 1 = ', transformed_initial_keypoints[0])
        print ('\tvote with geo constraint 2 = ', transformed_initial_keypoints[1])
        print ('\tvote with geo constraint 3 = ', transformed_initial_keypoints[2])

        print ('\tPost Horn keypoint offsets from gt')
        if gt_kpts is not None:
            kpt_err1 = np.sqrt((gt_kpts[0][0] - new_vote1[1])**2 + (gt_kpts[0][1] - new_vote1[2])**2 + (gt_kpts[0][2] - new_vote1[3])**2)
            kpt_err2 = np.sqrt((gt_kpts[1][0] - new_vote2[1])**2 + (gt_kpts[1][1] - new_vote2[2])**2 + (gt_kpts[1][2] - new_vote2[3])**2)
            kpt_err3 = np.sqrt((gt_kpts[2][0] - new_vote3[1])**2 + (gt_kpts[2][1] - new_vote3[2])**2 + (gt_kpts[2][2] - new_vote3[3])**2)
        
        print ('\tkpt1 error = ', kpt_err1)
        print ('\tkpt2 error = ', kpt_err2)
        print ('\tkpt3 error = ', kpt_err3)
        print ('\tAverage error = ', (kpt_err1 + kpt_err2 + kpt_err3)/3.0)


    # epsilon = epsilon-0.1


    xyz_inliers1, radial_list_inliers1 = accumulate_inliers(xyz1_mm, radial_list1_mm, len(xyz1), new_vote1, epsilon)
    xyz_inliers2, radial_list_inliers2 = accumulate_inliers(xyz2_mm, radial_list2_mm, len(xyz2), new_vote2, epsilon)
    xyz_inliers3, radial_list_inliers3 = accumulate_inliers(xyz3_mm, radial_list3_mm, len(xyz3), new_vote3, epsilon)

    if len(xyz_inliers1) >= 4:
        second_vote1 = random_center_est(xyz_inliers1, radial_list_inliers1, epsilon, iterations)
        kpt1 = np.array([second_vote1[1], second_vote1[2], second_vote1[3]])
    else:
        kpt1 = new_vote1[1:4]

    if len(xyz_inliers2) >= 4:
        second_vote2 = random_center_est(xyz_inliers2, radial_list_inliers2, epsilon, iterations)
        kpt2 = np.array([second_vote2[1], second_vote2[2], second_vote2[3]])
    else:
        kpt2 = new_vote2[1:4]

    if len(xyz_inliers3) >= 4:
        second_vote3 = random_center_est(xyz_inliers3, radial_list_inliers3, epsilon, iterations)
        kpt3 = np.array([second_vote3[1], second_vote3[2], second_vote3[3]])
    else:
        kpt3 = new_vote3[1:4]

    if debug:
        print ('\tSecond vote 1 = ', second_vote1)
        print ('\tSecond vote 2 = ', second_vote2)
        print ('\tSecond vote 3 = ', second_vote3)
        print ()
        print ('\tFinal keypoint offsets from gt')
        
        if gt_kpts is not None:
            kpt_err1 = np.sqrt((gt_kpts[0][0] - kpt1[0])**2 + (gt_kpts[0][1] - kpt1[1])**2 + (gt_kpts[0][2] - kpt1[2])**2)
            kpt_err2 = np.sqrt((gt_kpts[1][0] - kpt2[0])**2 + (gt_kpts[1][1] - kpt2[1])**2 + (gt_kpts[1][2] - kpt2[2])**2)
            kpt_err3 = np.sqrt((gt_kpts[2][0] - kpt3[0])**2 + (gt_kpts[2][1] - kpt3[1])**2 + (gt_kpts[2][2] - kpt3[2])**2)

        print ('\tkpt1 error = ', kpt_err1)
        print ('\tkpt2 error = ', kpt_err2)
        print ('\tkpt3 error = ', kpt_err3)
        print ('\tAverage error = ', (kpt_err1 + kpt_err2 + kpt_err3)/3.0)
        wait = input('Press Enter to continue')
        print ('='*50)

    pre_refinement_kpts = np.array([new_vote1[1:4], new_vote2[1:4], new_vote3[1:4]])

    refined_kpts = np.array([kpt1, kpt2, kpt3])

    return refined_kpts, pre_refinement_kpts
    

def RANSAC_refine(xyz, radial_list, iterations, epsilon):
    assert isinstance(iterations, int), 'iterations must be an integer'
   
    len_xyz = len(xyz)

    xyz_mm = xyz*1000
    radial_list_mm = radial_list*100

    # best_vote, xyz_inliers, radial_list_inliers = random_center_est_and_inliers(xyz_mm, radial_list_mm, epsilon, iterations, debug)

    best_vote = random_center_est(xyz_mm, radial_list_mm, epsilon, iterations)

    # if best_vote[0] == len_xyz:
        # center = center_est(xyz_mm, radial_list_mm)
        # center = np.array([center[0], center[1], center[2]])
        # center = center.astype("float64")
        # return center, len_xyz

    xyz_inliers, radial_list_inliers = accumulate_inliers(xyz_mm, radial_list_mm, len_xyz, best_vote, epsilon, early_stop=best_vote[0])

    
    if len(xyz_inliers) >= 4:
        center = center_est(xyz_inliers, radial_list_inliers)

        center = np.array([center[0], center[1], center[2]])

        center = center.astype("float64")

        return center, len(xyz_inliers)
    else:
        center = np.array([best_vote[1], best_vote[2], best_vote[3]])

        center = center.astype("float64")

        return center, 0
    


def main():

    #   generate random point, within unit sphere
    c = [random.random(), random.random(), random.random()]
    print('center = ', c)

    N = 100
    point_list = []
    radius_list = []
    for i in range(N):
        p = [random.random(), random.random(), random.random()]
        r = np.sqrt((p[0]-c[0])*(p[0]-c[0]) \
                    + (p[1]-c[1])*(p[1]-c[1]) \
                    + (p[2]-c[2])*(p[2]-c[2]))
        point_list += [p]
        radius_list += [r]
    # print(point_list)
    # print(radius_list)

    x, y, z = center_est(point_list, radius_list)
    print('estimated center = ', x, y, z, ', mse = ', np.sqrt((x-c[0])*(x-c[0]) + (y-c[1])*(y-c[1]) + (z-c[2])*(z-c[2])))

    eps = 5e-2
    for i in range(len(radius_list)):
        radius_list[i] += eps*random.random() * (-1)**random.randint(0,1)

    x, y, z = center_est(point_list, radius_list)
    print('estimated center = ', x, y, z, ', mse = ', np.sqrt((x-c[0])*(x-c[0]) + (y-c[1])*(y-c[1]) + (z-c[2])*(z-c[2])))


if __name__ == '__main__':
    print('running spherest ...')

    main()