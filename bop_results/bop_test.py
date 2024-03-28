from horn import HornPoseFitting
import numpy as np
from PIL import Image
import os
import open3d as o3d
import time
from ransac import RANSAC_refine
import pandas as pd
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt

lm_cls_names = ['ape', 'benchvise', 'cam', 'can', 'cat', 'duck', 'driller', 'eggbox', 'glue', 'holepuncher','iron','lamp','phone']

lmo_cls_names = ['ape', 'can', 'cat', 'duck', 'driller',  'eggbox', 'glue', 'holepuncher']

lm_cls_ids = {
    'ape': 1,
    'benchvise': 2,
    'cam': 4,
    'can': 5,
    'cat': 6,
    'driller': 8,
    'duck': 9,
    'eggbox': 10,
    'glue': 11,
    'holepuncher': 12,
    'iron': 13,
    'lamp': 14,
    'phone': 15
}

add_threshold = {
                  'eggbox': 0.019735770122546523,
                  'ape': 0.01421240983190395,
                  'cat': 0.018594838977253875,
                  'cam': 0.02222763033276377,
                  'duck': 0.015569664208967385,
                  'glue': 0.01930723067998101,
                  'can': 0.028415044264086586,
                  'driller': 0.031877906042,
                  'holepuncher': 0.019606109985,
                  'benchvise': .033091264970068,
                  'iron':.03172344425531,
                  'lamp':.03165980764376,
                  'phone':.02543407135792}



lm_syms = ['eggbox', 'glue']


linemod_K = np.array([[572.4114, 0., 325.2611],
                  [0., 573.57043, 242.04899],
                  [0., 0., 1.]])

#IO function from PVNet
def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    #pointc->actual scene
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    actual_xyz=xyz
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy,actual_xyz

def rgbd_to_point_cloud(K, depth):
    vs, us = depth.nonzero()
    zs = depth[vs, us]
    #print(zs.min())
    #print(zs.max())
    xs = ((us - K[0, 2]) * zs) / float(K[0, 0])
    ys = ((vs - K[1, 2]) * zs) / float(K[1, 1])
    pts = np.array([xs, ys, zs]).T
    return pts
    
def rgbd_to_color_point_cloud(K, depth, rgb):
    vs, us = depth.nonzero()
    zs = depth[vs, us]
    r = rgb[vs,us,0]
    g = rgb[vs,us,1]
    b = rgb[vs,us,2]
    #print(zs.min())
    #print(zs.max())
    xs = ((us - K[0, 2]) * zs) / float(K[0, 0])
    ys = ((vs - K[1, 2]) * zs) / float(K[1, 1])
    pts = np.array([xs, ys, zs, r, g, b]).T
    return pts

def rgbd_to_point_cloud_no_depth(K, depth):
    vs, us = depth.nonzero()
    zs = depth[vs, us]
    zs_min = zs.min()
    zs_max = zs.max()
    iter_range = int(zs_max*1000)+1-int(zs_min*1000)
    pts=[]
    for i in range(iter_range):
        if(i%1==0):
            z_tmp = np.empty(zs.shape) 
            z_tmp.fill(zs_min+i*0.001)
            xs = ((us - K[0, 2]) * z_tmp) / float(K[0, 0])
            ys = ((vs - K[1, 2]) * z_tmp) / float(K[1, 1])
            if(i == 0):
                pts = np.expand_dims(np.array([xs, ys, z_tmp]).T, axis=0)
                #print(pts.shape)
            else:
                pts = np.append(pts, np.expand_dims(np.array([xs, ys, z_tmp]).T, axis=0), axis=0)
                #print(pts.shape)
    print(pts.shape)
    return pts

#@jit(nopython=True)
def coords_inside_image(rr, cc, shape, val=None):
    """
    Modified based on https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/draw/draw.py#L484-L544
    Return the coordinates inside an image of a given shape.
    Parameters
    ----------
    rr, cc : (N,) ndarray of int
        Indices of pixels.
    shape : tuple
        Image shape which is used to determine the maximum extent of output
        pixel coordinates.  Must be at least length 2. Only the first two values
        are used to determine the extent of the input image.
    val : (N, D) ndarray of float, optional
        Values of pixels at coordinates ``[rr, cc]``.
    Returns
    -------
    rr, cc : (M,) array of int
        Row and column indices of valid pixels (i.e. those inside `shape`).
    val : (M, D) array of float, optional
        Values at `rr, cc`. Returned only if `val` is given as input.
    """
    mask = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
    if val is None:
        return rr[mask], cc[mask]
    else:
        return rr[mask], cc[mask], val[mask]



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


def refine_estimation_with_icp(source_points, target_points, initial_transformation, obj):
    """
    Refines the estimation of rotation and translation between source and target point sets using ICP.

    Parameters:
    - source_points: np.ndarray of shape (N, 3), representing the source keypoints.
    - target_points: np.ndarray of shape (N, 3), representing the target keypoints.
    - initial_transformation: np.ndarray of shape (4, 4), initial guess of the transformation matrix.

    Returns:
    - refined_transformation: np.ndarray of shape (4, 4), refined transformation matrix after ICP.
    """
    # Convert numpy arrays to Open3D point clouds
    source_pc = o3d.geometry.PointCloud()
    target_pc = o3d.geometry.PointCloud()
    source_pc.points = o3d.utility.Vector3dVector(source_points)
    target_pc.points = o3d.utility.Vector3dVector(target_points)

    # Run ICP refinement
    if np.asarray(target_pc.compute_point_cloud_distance(source_pc)).size > 0:
        threshold = np.asarray(target_pc.compute_point_cloud_distance(source_pc)).min()
    else:
        print ('No points found')
        return initial_transformation
    
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=add_threshold[obj],
                                                                  relative_rmse=add_threshold[obj],
                                                                    max_iteration=30)
    
    result = o3d.pipelines.registration.registration_icp(
        source_pc, target_pc, threshold, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria)

    return result.transformation



def estimate_6d_pose_lmo(opts):
    print ('Estimating 6D poses for LMO dataset and saving to CSV file...')
    horn = HornPoseFitting()
    scene_id = 2

    iterations = opts.iterations
    epsilon = opts.epsilon
    early_stop = opts.early_stop

    print ('Iterations: ', iterations)
    print ('Epsilon: ', epsilon)
    print ('Early Stop: ', early_stop)
    print ('ICP: ', opts.icp)
    
    with open(opts.out_dir + 'parameters.txt', 'w') as f:
        f.write(f"Iterations: {iterations}\n")
        f.write(f"Epsilon: {epsilon}\n")
        f.write(f"Early Stop: {early_stop}\n")

    debug = opts.verbose

    occ_path = opts.root_dataset + 'OCCLUSION_LINEMOD/'
    base_path = opts.root_dataset + 'LINEMOD/'

    general_counter = 0

    objs = []
    point_clouds = []
    obj_keypoints = []
    obj_max_radii = []

    for obj in lmo_cls_names:
        objs.append(obj)
        pcd_load = o3d.io.read_point_cloud(base_path + obj + '/' + obj + '.ply')
        xyz_load = np.asarray(pcd_load.points)
        point_clouds.append(xyz_load)
        keypoints = np.load(base_path + obj + '/Outside9.npy')
        obj_keypoints.append(keypoints)

        max_radii_dm = np.zeros(3)
        for i in range(3):
            dsitances = ((xyz_load[:,0]-keypoints[i+1,0])**2
                 +(xyz_load[:,1]-keypoints[i+1,1])**2
                +(xyz_load[:,2]-keypoints[i+1,2])**2)**0.5
            max_radii_dm[i] = dsitances.max()*10

        obj_max_radii.append(max_radii_dm)

    jpg_path = occ_path + 'RGB-D/rgb_noseg/'
    depth_path = occ_path + 'RGB-D/depth_noseg/'

    file_count = len(os.listdir(jpg_path))

    csv_path = opts.out_dir + 'estimated_data.csv'
    if os.path.exists(csv_path):
        os.remove(csv_path)


    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = ['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


    for filename in tqdm(os.listdir(jpg_path), disable=debug, total=file_count, desc='Processing images', position=0, leave=True, unit='image'):

        stripped_filename = filename.split('_')[1].split('.')[0]
        img_id = int(stripped_filename)
        if debug:
            print ('Processing image: ', img_id, ' ', general_counter, '/', file_count)

        condition = True
        rows = []
        start_time = time.time()

        depth = read_depth(depth_path + 'depth_' + stripped_filename + '.png')
        depth = np.array(depth, dtype=np.float64)

        for obj in lmo_cls_names:
            obj_id = lm_cls_ids[obj]
            if debug:
                print ('Processing object: ', obj, ' ', obj_id)

            condition = (os.path.isfile(occ_path+"blender_poses/"+obj+'/pose'+str(img_id)+'.npy')) and (
                            os.path.isfile(occ_path+"estRadialMap/"+obj+"/Out_pt1_dm/_"+str(img_id).zfill(5)+".npy")) and (
                                os.path.isfile(occ_path+"estRadialMap/"+obj+"/Out_pt2_dm/_"+str(img_id).zfill(5)+".npy")) and (
                                    os.path.isfile(occ_path+"estRadialMap/"+obj+"/Out_pt3_dm/_"+str(img_id).zfill(5)+".npy"))
            
            if not condition:
                continue

            xyz_load = point_clouds[lmo_cls_names.index(obj)]
            keypoints = obj_keypoints[lmo_cls_names.index(obj)]
            max_radii = obj_max_radii[lmo_cls_names.index(obj)]

            RTGT = np.load(occ_path + 'blender_poses/' + obj + '/pose' + str(img_id) + '.npy')

            transformed_gt_center = (np.dot(keypoints, RTGT[:, :3].T) + RTGT[:, 3:].T)*1000
        
            estimated_kpts = np.zeros((3,3))

            for i in range(3):
                keypoint_count = i+1
            
                radial_out = np.load(os.path.join(occ_path, 'estRadialMap', obj, 'Out_pt' + str(keypoint_count) + '_dm', '_'+str(int(os.path.splitext(filename)[0][6:])).zfill(5)+'.npy'))

                radial_out = np.where(radial_out<=max_radii[i], radial_out, 0)

                sem_out = np.where(radial_out>0, 1, 0)

                depth_map = depth*sem_out

                if radial_out.max() == 0:
                    continue

                pixel_coor = np.where(sem_out==1)

                radial_list = radial_out[pixel_coor]

                xyz_mm = rgbd_to_point_cloud(linemod_K, depth_map)

                xyz = xyz_mm/1000

                tic = time.time()
                center_mm_s, _ = RANSAC_refine(xyz, radial_list, iterations, epsilon, early_stop)
                toc = time.time()
        
                offset = ((transformed_gt_center[i+1,0]-center_mm_s[0])**2
                        +(transformed_gt_center[i+1,1]-center_mm_s[1])**2
                            +(transformed_gt_center[i+1,2]-center_mm_s[2])**2)**0.5
                
                if debug:
                    print ('\tKeypoint ', str(keypoint_count), ' Offset: ', offset, 'mm')
                    print ('\tRANSAC time: ', toc-tic, 's')

                estimated_kpts[i] = center_mm_s


            tic = time.time()
            kpts = keypoints[1:4, :]*1000
            RT = np.zeros((4,4))
            horn.lmshorn(kpts, estimated_kpts, 3, RT)

            RTGT_mm = RTGT
            RTGT_mm[:, 3] = RTGT_mm[:, 3]*1000
            toc = time.time()

            if debug:
                print ('Horn time: ', toc-tic, 's')
                print ('GT Rotation:\n', RTGT_mm[0:3, 0:3])
                print ('GT Translation:\n', RTGT_mm[0:3, 3])
                print ('Estimated Rotation:\n', RT[0:3, 0:3])
                print ('Estimated Translation:\n', RT[0:3, 3])

            _, xyz_load_transformed=project(xyz_load*1000, linemod_K, RTGT_mm)
            _, xyz_load_est_transformed=project(xyz_load*1000, linemod_K, RT[0:3,:])

            if opts.icp:
                tic = time.time()
                refined_RT = refine_estimation_with_icp(xyz_load_transformed, xyz_load_est_transformed, RT, obj)
                toc = time.time()
                if debug:
                    print ('ICP time: ', toc-tic, 's')
            else:
                refined_RT = RT

            rows.append({
                'scene_id': scene_id,
                'im_id': img_id,
                'obj_id': obj_id,
                'score': 1,
                'R': ' '.join(map(str, refined_RT[0:3, 0:3].flatten())),
                't': ' '.join(map(str, refined_RT[0:3, 3])),
            })

            if debug:
                print ('Refined Rotation: \n', refined_RT[0:3, 0:3])
                print ('Refined Translation: \n', refined_RT[0:3, 3])
                

        end_time = time.time()

        elapsed_time = end_time - start_time
        if debug:
            print('Elapsed time: ', elapsed_time, 's')
            print('Writing to CSV file for image: ', img_id)

        if len (rows) > 0:
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                for row in rows:
                    row['time'] = elapsed_time
                    writer.writerow(row)

        general_counter += 1

        if debug:
            wait = input("PRESS ENTER TO CONTINUE.")
            print ('----------------------------------------------\n')



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dataset',
                    type=str,
                    default='D:/')
 
    
    parser.add_argument('--out_dir',
                        type=str,
                        default='logs/test1/')
    
    parser.add_argument('--verbose',
                        type=bool,
                        default=False)
    
    parser.add_argument('--iterations',
                        type=int,
                        default=100)
    
    parser.add_argument('--epsilon',
                        type=float,
                        default=0.05)

    parser.add_argument('--early_stop',
                        type=int,
                        default=None)
    
    parser.add_argument('--icp',
                        type=bool,
                        default=True)
    

    opts = parser.parse_args()   

    if os.path.exists(opts.out_dir) == False:
        os.makedirs(opts.out_dir)

    print ('out_dir: ', opts.out_dir)

    estimate_6d_pose_lmo(opts)
