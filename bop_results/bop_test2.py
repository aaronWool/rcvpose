from horn import HornPoseFitting
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import open3d as o3d
import time
from ransac import RANSAC, RANSAC_refine, center_est
from numba import prange
from numba import jit
import math
import csv
#import h5py
from sklearn import metrics
import scipy


lm_cls_names = ['ape', 'benchvise', 'cam', 'can', 'cat', 'duck', 'driller', 'eggbox', 'glue', 'holepuncher','iron','lamp','phone']

lmo_cls_names = ['ape', 'can', 'cat', 'duck', 'driller',  'eggbox', 'glue', 'holepuncher']

ycb_cls_names={1:'002_master_chef_can',
           2:'003_cracker_box',
           3:'004_sugar_box',
           4:'005_tomato_soup_can',
           5:'006_mustard_bottle',
           6:'007_tuna_fish_can',
           7:'008_pudding_box',
           8:'009_gelatin_box',
           9:'010_potted_meat_can',
           10:'011_banana',
           11:'019_pitcher_base',
           12:'021_bleach_cleanser',
           13:'024_bowl',
           14:'025_mug',
           15:'035_power_drill',
           16:'036_wood_block',
           17:'037_scissors',
           18:'040_large_marker',
           19:'051_large_clamp',
           20:'052_extra_large_clamp',
           21:'061_foam_brick'}
lm_syms = ['eggbox', 'glue']
ycb_syms = ['024_bowl','036_wood_block','051_large_clamp','052_extra_large_clamp','061_foam_brick']
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


@jit(nopython=True, parallel=True)
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

def Accumulator_3D(xyz, radial_list):
    acc_unit = 5
    # unit 5mm 
    xyz_mm = xyz*1000/acc_unit #point cloud is in meter

    #print(xyz_mm)
    
    #recenter the point cloud
    x_mean_mm = np.mean(xyz_mm[:,0])
    y_mean_mm = np.mean(xyz_mm[:,1])
    z_mean_mm = np.mean(xyz_mm[:,2])
    xyz_mm[:,0] -= x_mean_mm
    xyz_mm[:,1] -= y_mean_mm
    xyz_mm[:,2] -= z_mean_mm
    
    radial_list_mm = radial_list*100/acc_unit  #radius map is in decimetre for training purpose
    
    xyz_mm_min = xyz_mm.min()
    xyz_mm_max = xyz_mm.max()
    radius_max = radial_list_mm.max()
    
    zero_boundary = int(xyz_mm_min-radius_max)+1
    
    if(zero_boundary<0):
        xyz_mm -= zero_boundary
        #length of 3D vote map 
    length = int(xyz_mm.max())
    
    VoteMap_3D = np.zeros((length+int(radius_max),length+int(radius_max),length+int(radius_max)))
    tic = time.perf_counter()
    VoteMap_3D = fast_for(xyz_mm,radial_list_mm,VoteMap_3D)
    toc = time.perf_counter()
                        
    center = np.argwhere(VoteMap_3D==VoteMap_3D.max())
   # print("debug center raw: ",center)
    center = center.astype("float64")
    if(zero_boundary<0):
        center = center+zero_boundary
        
    #return to global coordinate
    center[0,0] = (center[0,0]+x_mean_mm+0.5)*acc_unit
    center[0,1] = (center[0,1]+y_mean_mm+0.5)*acc_unit
    center[0,2] = (center[0,2]+z_mean_mm+0.5)*acc_unit
    
    #center = center*acc_unit+((3**0.5)/2)

    return center

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


def estimate_6d_pose_lmo(opts):

    csv_path = opts.out_dir + 'estimated_data.csv'
    if os.path.exists(csv_path):
        os.remove(csv_path)

    

    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = ['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


    horn = HornPoseFitting()
    for class_name in lmo_cls_names:
        # wait = input("PRESS ENTER TO CONTINUE.")
        file_skip_counter = 0
        print(class_name)
        rootPath = opts.root_dataset+'OCCLUSION_LINEMOD/'
        general_counter = 0
        valid_counter = 0
        
        #counters
        bf_icp = 0
        af_icp = 0
        #h5 save keypoints
        filenameList=[]
        model_list=[]
        offsets = []


        test_list = open('test.txt', 'r').readlines()
        for i in range(len(test_list)):
            test_list[i] = int(test_list[i].split('/')[1].split('_')[0])
        test_list = list(set(test_list))
    
        
        pcd_load = o3d.io.read_point_cloud(opts.root_dataset+"LINEMOD/"+class_name+"/"+class_name+".ply")
        xyz_load = np.asarray(pcd_load.points)

        keypoints=np.load(opts.root_dataset+"LINEMOD/"+class_name+"/"+"Outside9.npy")

        #threshold of radii maximum limits
        max_radii_dm = np.zeros(3)
        for i in range(3):
            dsitances = ((xyz_load[:,0]-keypoints[i+1,0])**2
                 +(xyz_load[:,1]-keypoints[i+1,1])**2
                +(xyz_load[:,2]-keypoints[i+1,2])**2)**0.5
            max_radii_dm[i] = dsitances.max()*10

        
        jpgPath = rootPath + "RGB-D/rgb_noseg/"
        depthPath = rootPath + "RGB-D/depth_noseg/"
        wrong_samples = 0
        
        for filename in os.listdir(jpgPath):
            time_start = time.time()
            #filename = 'color_00274.png'
            #print(os.path.splitext(filename)[0][6:].zfill(6))
            img_id = int(os.path.splitext(filename)[0][6:])

            if int(os.path.splitext(filename)[0][6:]) not in test_list:
                continue

            print (img_id)

            
            #model_path = "ape_pt0_syn18.pth.tar"
            ptsList=[]
            iter_count = 0
            keypoint_count=1

            #GTDepthPath = rootPath+'GeneratedDepth/'+class_name+'/'
            estimated_kpts = np.zeros((3,3))
            xyz_load_transformed = []
            RTGT=[]
            condition = True
            #wrong_samples = 0


            for keypoint in keypoints:
                keypoint = keypoints[keypoint_count]
                #model_path = opts.model_dir + class_name+"_pt"+str(keypoint_count)+".pth.tar"
                true_center = keypoint
                #keypoint = keypoints[1]            
                #filename = "color_00076.png"
                if filename.endswith(".png"):
                    #print(filename)
                    #get the transformed gt center
                    if opts.using_ckpts:
                        condition = (os.path.isfile(rootPath+"blender_poses/"+class_name+
                                      "/pose"+str(int(os.path.splitext(filename)[0][6:]))+'.npy'))
                    else:
                        condition = (os.path.isfile(rootPath+"blender_poses/"+class_name+
                                      "/pose"+str(int(os.path.splitext(filename)[0][6:]))+'.npy')) and (
                                          os.path.isfile(os.path.join(rootPath, 'estRadialMap', class_name, 
                                             'Out_pt'+str(keypoint_count)+'_dm', 
                                             '_'+str(int(os.path.splitext(filename)[0][6:])).zfill(5)+'.npy')))
                    if not condition:
                        file_skip_counter += 1
                    if condition:
                    #and (
                    #                      os.path.isfile(os.path.join(rootPath, 'estRadialMap', class_name, 
                    #                         'Out_pt'+str(keypoint_count)+'_dm', 
                    #                         '_'+str(int(os.path.splitext(filename)[0][6:])).zfill(5)+'.npy'))):                
                        RTGT = np.load(rootPath+"blender_poses/"+class_name+
                                       "/pose"+str(int(os.path.splitext(filename)[0][6:]))+'.npy')
                        #print(RT)

                        input_path = jpgPath +filename
                        depth_map = Image.open(depthPath+'depth_'+os.path.splitext(filename)[0][6:].zfill(5)+'.png')
                        depth_map = np.array(depth_map, dtype=np.float64)
                        #depth_map = depth_map/1000
                        if opts.using_ckpts:
                            print('womp womp')
                        else:
                            #print('inside else')
                            radial_out = np.load(
                                os.path.join(rootPath, 'estRadialMap', class_name, 
                                             'Out_pt'+str(keypoint_count)+'_dm', 
                                             '_'+str(int(os.path.splitext(filename)[0][6:])).zfill(5)+'.npy'))
                            #plt.imshow(radial_out)
                            #plt.show()
                            radial_out = np.where(radial_out<=max_radii_dm[keypoint_count-1],radial_out,0)
                            sem_out = np.where(radial_out>0,1,0)
                            depth_map = depth_map*sem_out
                        #plt.imshow(sem_out)
                        #plt.show()

                        mean = 0.84241277810665
                        std = 0.12497967663932731
                        
                        if radial_out.max()!=0:
                            pixel_coor = np.where(sem_out==1)

                            #if opts.using_ckpts:
                            radial_list = radial_out[pixel_coor]
                            xyz_mm = rgbd_to_point_cloud(linemod_K,depth_map)
                            xyz = xyz_mm/1000
                           # dump, xyz_load_transformed=project(xyz_load, linemod_K, RT)
                      
                        
                            center_mm_s, _ = RANSAC_refine(xyz, radial_list, 1000, 0.04)
                       


                            #pre_center_off_mm = math.inf
                            transformed_gt_center_mm = (np.dot(keypoints, RTGT[:, :3].T) + RTGT[:, 3:].T)*1000

                            transformed_gt_center_mm = transformed_gt_center_mm[keypoint_count]

                            estimated_center_mm = center_mm_s
                            #estimated_center_mm = transformed_gt_center_mm[0]
                            center_off_mm = ((transformed_gt_center_mm[0]-estimated_center_mm[0])**2+
                                            (transformed_gt_center_mm[1]-estimated_center_mm[1])**2+
                                            (transformed_gt_center_mm[2]-estimated_center_mm[2])**2)**0.5     
                            
                            if center_off_mm > 1000:
                                ransac_error = center_off_mm
                                print ('RANSAC failure for image: ', filename)
                                print ('Estimated center: ', estimated_center_mm)
                                print ('GT center: ', transformed_gt_center_mm)
                                print ('Offset: ', center_off_mm)
                                center_mm_s = Accumulator_3D(xyz, radial_list)[0]
                                print ('Accumulator center: ', center_mm_s)
                                center_off_mm = ((transformed_gt_center_mm[0]-center_mm_s[0])**2+
                                            (transformed_gt_center_mm[1]-center_mm_s[1])**2+
                                            (transformed_gt_center_mm[2]-center_mm_s[2])**2)**0.5
                                print ('Accumulator offset: ', center_off_mm)
                                if center_off_mm < ransac_error:
                                    estimated_center_mm = center_mm_s
                                    print ('Accumulator center chosen')
                                # plt.imshow(radial_out)
                                # plt.show()

                            offsets.append(center_off_mm)
                            

                            estimated_kpts[keypoint_count-1] = estimated_center_mm

                        keypoint_count+=1   
                        if keypoint_count > 3:
                            break 
            if condition:      
                #print(filename)     
                kpts = keypoints[1:4,:]*1000
                RT = np.zeros((4, 4))
                horn.lmshorn(kpts, estimated_kpts, 3, RT)
                RTGT_mm = RTGT
                RTGT_mm[:,3] = RTGT_mm[:,3]*1000
                dump, xyz_load_transformed=project(xyz_load*1000, linemod_K, RTGT_mm)
                dump, xyz_load_est_transformed=project(xyz_load*1000, linemod_K, RT[0:3,:])
                if opts.demo_mode:
                    input_image = np.asarray(Image.open(input_path).convert('RGB'))
                    for coor in dump:
                        input_image[int(coor[1]),int(coor[0])] = [255,0,0]
                    plt.imshow(input_image)
                    plt.show()
                sceneGT = o3d.geometry.PointCloud()
                sceneEst = o3d.geometry.PointCloud()
                sceneGT.points=o3d.utility.Vector3dVector(xyz_load_transformed)
                sceneEst.points=o3d.utility.Vector3dVector(xyz_load_est_transformed)
                sceneGT.paint_uniform_color(np.array([0,0,1]))
                sceneEst.paint_uniform_color(np.array([1,0,0]))
                if opts.demo_mode:
                    o3d.visualization.draw_geometries([sceneGT, sceneEst],window_name='gt vs est before icp')
                #print('ADD(s) point distance before ICP: ', distance)
                if class_name in lm_syms:
                    if np.asarray(sceneGT.compute_point_cloud_distance(sceneEst)).size>0:
                        min_distance = np.asarray(sceneGT.compute_point_cloud_distance(sceneEst)).min()
                        if min_distance <= add_threshold[class_name]*1000:
                            bf_icp+=1
                        threshold = min_distance
                    else:
                        threshold = 5
                else:
                    if np.asarray(sceneGT.compute_point_cloud_distance(sceneEst)).size>0:
                        distance = np.asarray(sceneGT.compute_point_cloud_distance(sceneEst)).mean()
                        if distance <= add_threshold[class_name]*1000:
                            bf_icp+=1   
                        threshold = distance
                    else:
                        threshold = 5
                trans_init = np.asarray([[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 1, 0], 
                                        [0, 0, 0, 1]])

                criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness = add_threshold[class_name]*1000,
                                                                             relative_rmse = add_threshold[class_name]*1000,
                                                                             max_iteration=30)
                reg_p2p = o3d.pipelines.registration.registration_icp(
                    sceneGT, sceneEst,  threshold, trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    criteria)
                

                print ('Files Skipped: ', file_skip_counter)
                print ('Ground truth Rotation matrix:\n ', RTGT[0:3, 0:3])
                print ('Ground truth Translation matrix:\n ', RTGT[0:3, 3])
                print ('Estimated Rotation matrix:\n ', RT[0:3, 0:3])
                print ('Estimated Translation matrix:\n ', RT[0:3, 3])
                refined_RT = reg_p2p.transformation
                print ('Refined Rotation matrix:\n ', refined_RT[0:3, 0:3])
                print ('Refined Translation matrix:\n ', refined_RT[0:3, 3])
                # wait = input("PRESS ENTER TO CONTINUE.")


                sceneGT.transform(reg_p2p.transformation)
                if opts.demo_mode:
                    o3d.visualization.draw_geometries([sceneGT, sceneEst],window_name='gt vs est after icp')
                #print('ADD(s) point distance after ICP: ', distance)
                if class_name in lm_syms:
                    if np.asarray(sceneGT.compute_point_cloud_distance(sceneEst)).size>0:
                        min_distance = np.asarray(sceneGT.compute_point_cloud_distance(sceneEst)).min()
                        if min_distance <= add_threshold[class_name]*1000:
                           af_icp+=1
                else:
                    if np.asarray(sceneGT.compute_point_cloud_distance(sceneEst)).size>0:
                        distance = np.asarray(sceneGT.compute_point_cloud_distance(sceneEst)).mean()
                        if distance <= add_threshold[class_name]*1000:
                            af_icp+=1         


                obj_id = lm_cls_ids[class_name]

 

                with open(csv_path, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow({'scene_id': 2, 'im_id': img_id, 'obj_id': obj_id, 'score': 1, 'R': ' '.join(map(str, refined_RT[0:3, 0:3].flatten())), 't': ' '.join(map(str, refined_RT[0:3, 3].flatten())), 'time': '5'})
                                                          
            general_counter += 1
            #print(af_icp)
            if class_name in lm_syms:    
                print('Current ADDs of '+class_name+' before ICP: ', bf_icp/general_counter)
                print('Currnet ADDs of '+class_name+' after ICP: ', af_icp/general_counter) 
            else:
                print('Current ADD of '+class_name+' before ICP: ', bf_icp/general_counter)
                print('Current ADD of '+class_name+' after ICP: ', af_icp/general_counter)    
            print ('Current offset: ', np.mean(offsets))
            print ('Current std: ', np.std(offsets))
            print('Processed: ', round((general_counter/len(test_list))*100, 2), '%\n')
        if class_name in lm_syms:    
            print('ADDs of '+class_name+' before ICP: ', bf_icp/general_counter)
            print('ADDs of '+class_name+' after ICP: ', af_icp/general_counter) 
        else:
            print('ADD of '+class_name+' before ICP: ', bf_icp/general_counter)
            print('ADD of '+class_name+' after ICP: ', af_icp/general_counter)  

        with open(opts.out_dir + 'ADDs.txt', 'a') as f:
            f.write('ADDs of '+class_name+' before ICP: '+str(bf_icp/general_counter)+'\n')
            f.write('ADDs of '+class_name+' after ICP: '+str(af_icp/general_counter)+'\n')
            f.write('Average offset: '+str(np.mean(offsets))+'\n')
            f.write('Average Std of offset: '+str(np.std(offsets))+'\n')
            f.write('\n')       


if __name__ == "__main__":


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dataset',
                    type=str,
                    default='D:/')
    parser.add_argument('--model_dir',
                    type=str,
                    default='ckpts/')   
    parser.add_argument('--demo_mode',
                    type=bool,
                    default=False)  
    parser.add_argument('--using_ckpts',
                    type=bool,
                    default=False)

    parser.add_argument('--frontend',
                        type=str,
                        default='ransac',
                        choices=['accumulator_space', 'ransac', 'RANSAC', ])
    
    parser.add_argument('--out_dir',
                        type=str,
                        default='logs/lmo_bop/')


    
    opts = parser.parse_args()   

    if os.path.exists(opts.out_dir) == False:
        os.makedirs(opts.out_dir)

    print ('out_dir: ', opts.out_dir)

    
    if opts.frontend == 'ransac':
        opts.frontend = 'RANSAC'


    estimate_6d_pose_lmo(opts)

