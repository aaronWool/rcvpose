from numba import jit, prange
import numpy as np
import math 
import os


def RANSAC_3D(xyz, radial_list, epsilon, debug = False):
    return None



def RANSAC_debug(dataset, epsilon, iter):
    


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        help='Path to the dataset',
        default='dataset/LINEMOD/ape/'
    )
    parser.add_argument(
        '--epsilon', '-e',
        type=float,
        help='Epsilon',
        default=0.01
    )
    parser.add_argument(
        '--iter', '-i',
        type=int,
        help='Number of iterations',
        default=1000
    )

    opts = parser.parse_args()

    RANSAC_debug(opts.dataset, opts.epsilon, opts.iter)
    

