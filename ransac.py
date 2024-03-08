import numba
from numba import jit, prange, njit
import numpy as np
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
def random_center_est(xyz, radial_list, epsilon, iterations=25, debug=False):

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
    

# Iterate through all the data points and accumulate inliers
@njit(parallel=True)
def accumulate_inliers(xyz, radial_list, iterations, best_vote, epsilon):
   
    xyz_inliers = np.zeros((iterations, 3))  
    radial_list_inliers = np.zeros(iterations)
    inlier_count = 0

    indexes = np.arange(len(xyz))
    np.random.shuffle(indexes)

    for itr in prange(iterations):
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


def RANSAC(xyz, radial_list, iterations, epsilon, debug=False):
    xyz_mm = xyz*1000
    radial_list_mm = radial_list*100    

    best_vote = random_center_est(xyz_mm, radial_list_mm, epsilon, iterations, debug)

    center = np.array([best_vote[1], best_vote[2], best_vote[3]])

    center = center.astype("float64")
    
    return center

def RANSAC_refine(xyz, radial_list, iterations, epsilon, debug=False):

    pc_size = xyz.shape[0]

    xyz_mm = xyz*1000
    radial_list_mm = radial_list*100

    # best_vote, xyz_inliers, radial_list_inliers = random_center_est_and_inliers(xyz_mm, radial_list_mm, epsilon, iterations, debug)

    best_vote = random_center_est(xyz_mm, radial_list_mm, epsilon, iterations, debug)

    xyz_inliers, radial_list_inliers = accumulate_inliers(xyz_mm, radial_list_mm, pc_size, best_vote, epsilon)
    
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