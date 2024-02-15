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

# Refine Consensus with only inliers, uses best fitting point to determine center
@jit(nopython=True, parallel=True)
def refine_consensus(xyz, radial_list, iterations):
    
    xyz_len = len(xyz)
    n = xyz_len

    votes = np.zeros((iterations, 4))

    for itr in prange(iterations):
        index = np.random.randint(0, xyz_len, 4)

        point_list = xyz[index]

        radius_list = radial_list[index]
        
        x, y, z = centerest(point_list, radius_list)

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

def estimate_and_accumulate(xyz, radial_list, iterations=100, epsilon = 0.7, max_inliers=200, debug=False):
    # Get a random centerest
    current_epsilon = epsilon
    epsilon_increase_1 = 0
    best_vote = random_centerest(xyz, radial_list, iterations, epsilon=current_epsilon, debug=debug)

    # If no inliers, increase epsilon
    if best_vote[0] == 0:
        while best_vote[0] == 0:
            current_epsilon += 0.05
            epsilon_increase_1 += 1
            best_vote = random_centerest(xyz, radial_list, iterations, epsilon=current_epsilon, debug=debug)

    # Accumulate inliers
    current_epsilon = epsilon
    num_iterations = len(xyz)
    epsilon_increase_2 = 0

    xyz_inliers, radial_list_inliers = accumulate_inliers(xyz, radial_list, num_iterations , best_vote, current_epsilon, max_inliers=max_inliers)
    
    # If no inliers, increase epsilon
    if xyz_inliers == []:
        while xyz_inliers == []:
            epsilon_increase_2 += 1
            current_epsilon += 0.05
            xyz_inliers, radial_list_inliers = accumulate_inliers(xyz, radial_list, num_iterations, best_vote, current_epsilon, max_inliers=max_inliers)
    
    if debug and (epsilon_increase_1 > 0 or epsilon_increase_2 > 0):
        print('\t\tNumber of epsilon increases for first random centerest: ' + str(epsilon_increase_1))
        print('\t\tNumber of epsilon increases for first accumulate inliers: ' + str(epsilon_increase_2))

    return best_vote, xyz_inliers, radial_list_inliers


def ransac_4(xyz, radial_list, iterations=100, epsilon = 0.7, debug=False):
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
    
    max_inliers = 400
    
    current_epsilon = epsilon

    if debug:
        print ('\tNumber of initial data points: ', len(xyz_mm))

    best_vote, xyz_inliers, radial_list_inliers = estimate_and_accumulate(xyz_mm, radial_list_mm, iterations, epsilon, max_inliers=max_inliers, debug=debug)
    if len(xyz_inliers) > 200:
        best_vote, xyz_inliers, radial_list_inliers = estimate_and_accumulate(xyz_inliers, radial_list_inliers, iterations, epsilon=epsilon, max_inliers=int(max_inliers/2), debug=debug)

    
    # Final Center incase of less than 4 inliers
    center = np.array([best_vote[1], best_vote[2], best_vote[3]])

    if debug:
        print('\tNumber of inliers after estimation process: ' + str(len(xyz_inliers)))


    if len(xyz_inliers) > 4:
        best_vote = refine_consensus(xyz_inliers, radial_list_inliers, iterations)
        center = np.array([best_vote[1], best_vote[2], best_vote[3]])
    

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

    x, y, z = centerest(point_list, radius_list)
    print('estimated center = ', x, y, z, ', mse = ', np.sqrt((x-c[0])*(x-c[0]) + (y-c[1])*(y-c[1]) + (z-c[2])*(z-c[2])))

    eps = 5e-2
    for i in range(len(radius_list)):
        radius_list[i] += eps*random.random() * (-1)**random.randint(0,1)

    x, y, z = centerest(point_list, radius_list)
    print('estimated center = ', x, y, z, ', mse = ', np.sqrt((x-c[0])*(x-c[0]) + (y-c[1])*(y-c[1]) + (z-c[2])*(z-c[2])))


if __name__ == '__main__':
    print('running spherest ...')

    main()