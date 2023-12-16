from numba import jit, prange
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
    # O(A.size)
    U, S, Vh = np.linalg.svd(A)
    X = Vh[-1]
    X /= X[-1]

    return X[0], X[1], X[2]



@jit(nopython=True, parallel=True)
def random_centerest(xyz, radial_list, iterations, debug=False):
    n = len(xyz)
    votes = np.zeros((iterations, 4))

    for itr in prange(iterations):
        index = np.random.randint(0, n, 4)

        point_list = xyz[index]

        radius_list = radial_list[index]
        
        x, y, z = centerest(point_list, radius_list)

        error = 0

        for i in prange(n):
            p = xyz[i]
            r = radial_list[i]
            dist = np.sqrt((p[0]-x)*(p[0]-x) + (p[1]-y)*(p[1]-y) + (p[2]-z)*(p[2]-z))
            error += abs(dist-r)

        error /= len(xyz)

        votes[itr] = (error, x, y, z)

    
    sorted_votes = sorted(votes, key=lambda x: x[0])
    
    return sorted_votes 


def RANSAC_3D(xyz, radial_list, iterations=2000, debug=False):
    acc_unit = 5


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
    
    max_num_refinements = 20
    epsilon = 10
    center = np.array([0, 0, 0])

    for itr in range(max_num_refinements):     
        if len(radial_list_mm) < 4:
            if debug:
                print('Not enough points, breaking')
            break

        best_vote = random_centerest(xyz_mm, radial_list_mm, iterations, debug=debug)
        iterations = int(iterations/2)
        best_vote = best_vote[0]
    
        xyz_inliers = []
        radial_list_inliers = []
        num_iterations = int(len(xyz_mm)/2)
    
        for _ in range(num_iterations):
            i = random.randint(0, len(xyz_mm) - 1)
            p = xyz_mm[i]
            r = radial_list_mm[i]
            dist = np.sqrt((p[0] - best_vote[1]) ** 2 + (p[1] - best_vote[2]) ** 2 + (p[2] - best_vote[3]) ** 2)
            if abs(dist - r) < epsilon:
                xyz_inliers.append(p)
                radial_list_inliers.append(r)


        epsilon = int(epsilon/1.5)
        xyz_mm = np.array(xyz_inliers)
        radial_list_mm = np.array(radial_list_inliers)

        if len(radial_list_inliers) >= 4:
            center = centerest(xyz_inliers, radial_list_inliers)
            center = np.array([center[0], center[1], center[2]])

        elif len(radial_list_inliers) < 4:
            if debug:
                print('Not enough inliers, breaking')
            break
        
    
    center = center.astype("float64")

    if(zero_boundary<0):
        center = center+zero_boundary

    center[0] = (center[0]+x_mean_mm+0.5)*acc_unit
    center[1] = (center[1]+y_mean_mm+0.5)*acc_unit
    center[2] = (center[2]+z_mean_mm+0.5)*acc_unit
    
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