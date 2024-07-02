from util.horn import HornPoseFitting
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import open3d as o3d
import time
import math
#import h5py
from sklearn import metrics


# lm_cls_names = ['ape', 'benchvise', 'cam', 'can', 'cat', 'duck', 'driller', 'eggbox', 'glue', 'holepuncher','iron','lamp','phone']

lm_cls_names = ['ape']




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





depthList=[]

def estimate_6d_pose_lm(opts, mean_radius_mm, std_dev_mm):
    horn = HornPoseFitting()

    offsets = []


    for class_name in lm_cls_names:


        # print("Evaluation on ", class_name)
        rootPath = opts.root_dataset + "LINEMOD_ORIG/"+class_name+"/"
        rootpvPath = opts.root_dataset +class_name+"/"
        test_list = open(opts.root_dataset +class_name+"/" +"Split/train.txt","r").readlines()
        test_list = [ s.replace('\n', '') for s in test_list]
        test_list_len = len(test_list)
        #print(test_list)

        pcd_load = o3d.io.read_point_cloud(opts.root_dataset +class_name+"/"+class_name+".ply")

        #time consumption
        net_time = 0
        acc_time = 0
        general_counter = 0

        #counters
        bf_icp = 0
        af_icp = 0
        model_list=[]

        auc_threshold = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

        auc_adds_count = np.zeros((2,11))


        #h5 save keypoints
        #h5f = h5py.File(class_name+'PointPairsGT.h5','a')

        filenameList = []

        class_auc_adds_count = np.zeros((2,11))
        xyz_load = np.asarray(pcd_load.points)
        #print(xyz_load)


        #keypoints=np.load(opts.root_dataset + "LINEMOD/"+class_name+"/"+"Outside9.npy")
        #print(keypoints)

        keypoints=np.load(rootpvPath + 'Outside9.npy')

     
        #print(max_radii_dm)
        dataPath = rootpvPath + 'JPEGImages/'

        for filename in tqdm(os.listdir(dataPath), desc='Processing '+class_name, leave=False):
            RTGT = np.load(opts.root_dataset +class_name+"/pose/pose"+str(int(os.path.splitext(filename)[0]))+'.npy')
            estimated_kpts = np.zeros((3,3))
            #filename = '000810.jpg'
            #print("Evaluating ", filename)
            if filename.endswith(".jpg"):
                #print(os.path.splitext(filename)[0][5:].zfill(6))
                if os.path.splitext(filename)[0] in test_list:
                #if filename in test_list:
                    kpts_w_gaussian_noise = np.zeros((3,3))

                    transformed_gt_center_mm = (np.dot(keypoints, RTGT[:, :3].T) + RTGT[:, 3:].T)*1000
                    # print(transformed_gt_center_mm)

                    keypoint_count = 1
                    for keypoint in keypoints:
                        keypoint=keypoints[keypoint_count]

                        random_radius = np.random.normal(loc=mean_radius_mm, scale=std_dev_mm)
                        # print(random_radius)
                        # exit()
                        # radius = np.clip(radius, 0, max_value_mm)

                        theta = np.random.uniform(0, 2*np.pi)
                        phi = np.random.uniform(0, np.pi)

                        x = random_radius*np.sin(phi)*np.cos(theta)
                        y = random_radius*np.sin(phi)*np.sin(theta)
                        z = 0#random_radius*np.cos(phi)

                        random_vector = np.array([x, y, z])
                        keypoint_w_added_noise = transformed_gt_center_mm[keypoint_count] + random_vector

                        offset = np.linalg.norm(keypoint_w_added_noise-transformed_gt_center_mm[keypoint_count])
                        offsets.append(offset)

                        # print ('Random Vector: \t', random_vector, 'mm')
                        # print ('Orig Position: \t', transformed_gt_center_mm[keypoint_count], 'mm')
                        # print ('Noisy Position: \t', keypoint_w_added_noise, 'mm')
                        # print ('Offset: ', offset, 'mm')
                        # exit()
                        # wait = input("PRESS ENTER TO CONTINUE.")

                        kpts_w_gaussian_noise[keypoint_count-1] = keypoint_w_added_noise

                        keypoint_count += 1

                        if keypoint_count > 3:
                            break


                    kpts = keypoints[1:4,:]*1000
                    RT = np.zeros((4, 4))
                    horn.lmshorn(kpts, kpts_w_gaussian_noise, 3, RT)
                    # print(RT)
                    dump, xyz_load_est_transformed=project(xyz_load*1000, linemod_K, RT[0:3,:])
                    RTGT_mm = RTGT
                    RTGT_mm[:,3] = RTGT_mm[:,3]*1000

                    # print(RTGT_mm)
                    dump, xyz_load_transformed=project(xyz_load*1000, linemod_K, RTGT_mm)

                    #xyz_load_est_transformed = xyz_load_est_transformed*1000

                    sceneGT = o3d.geometry.PointCloud()
                    sceneEst = o3d.geometry.PointCloud()
                    sceneGT.points=o3d.utility.Vector3dVector(xyz_load_transformed)
                    sceneEst.points=o3d.utility.Vector3dVector(xyz_load_est_transformed)
                    sceneGT.paint_uniform_color(np.array([0,0,1]))
                    sceneEst.paint_uniform_color(np.array([1,0,0]))


                    min_distance = np.asarray(sceneGT.compute_point_cloud_distance(sceneEst)).min()
                    distance = np.asarray(sceneGT.compute_point_cloud_distance(sceneEst)).mean()
                    if distance <= add_threshold[class_name]*1000:
                        bf_icp+=1

                    scene = o3d.geometry.PointCloud()
                    scene.points = o3d.utility.Vector3dVector(xyz_load_est_transformed)
                    cad_model = o3d.geometry.PointCloud()
                    cad_model.points = o3d.utility.Vector3dVector(xyz_load*1000)
                    # trans_init = np.asarray([[1, 0, 0, 0],
                    #                         [0, 1, 0, 0],
                    #                         [0, 0, 1, 0],
                    #                         [0, 0, 0, 1]])
                    trans_init = RT
                    #if class_name in lm_syms:
                    #    threshold = min_distance
                    #else:
                    threshold = distance
                    criteria = o3d.pipelines.registration.ICPConvergenceCriteria()
                    reg_p2p = o3d.pipelines.registration.registration_icp(
                        cad_model, scene, threshold, trans_init,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                        criteria)
                    cad_model.transform(reg_p2p.transformation)

                    distance = np.asarray(sceneGT.compute_point_cloud_distance(cad_model)).mean()
                    if distance <= add_threshold[class_name]*1000:
                        af_icp+=1
                    general_counter += 1
                    # print('Current ADD\(s\) of '+class_name+' before ICP: ', bf_icp/general_counter)
                    # print('Currnet ADD\(s\) of '+class_name+' after ICP: ', af_icp/general_counter)
                
                    # if class_name in lm_syms:
                    #     if min_distance <= add_threshold[class_name]*1000:
                    #         af_icp+=1
                    # else:
                    #     if distance <= add_threshold[class_name]*1000:
                    #         af_icp+=1

                    # i = 0
                    # for threshold in auc_threshold:
                    #     if class_name in lm_syms:
                    #         if min_distance <= threshold*1000:
                    #             auc_adds_count[1, i] += 1
                    #             class_auc_adds_count [1, i] += 1
                    #     else:
                    #         if distance <= threshold*1000:
                    #             auc_adds_count[1, i] += 1
                    #             class_auc_adds_count[1, i] += 1
                    #     i += 1

                    # general_counter += 1

                    # print('Mean Offset: ', np.mean(offsets))
                    # print('Current ADD\(s\) of '+class_name+' before ICP: ', bf_icp/general_counter)
                    # print('Currnet ADD\(s\) of '+class_name+' after ICP: ', af_icp/general_counter)
                    # print('Current AUC of ' + class_name + ' before ICP: ', metrics.auc(auc_threshold, class_auc_adds_count[0]/general_counter)/0.1)
                    # print('Current AUC of ' + class_name + ' after ICP: ', metrics.auc(auc_threshold, class_auc_adds_count[1]/general_counter)/0.1)
                    # print('Processed: ', round((general_counter/test_list_len)*100, 2), '%\n')
                    # break

                    # if general_counter > 20:
                        # break

                    if general_counter/test_list_len == 1:
                        break



        #os.system("pause")

        print('='*20)
        print ('Gaussian Mean: ', mean_radius_mm, 'mm')
        print ('Gaussian Std Dev: ', std_dev_mm, 'mm')
        print('ADDs of '+class_name+' before ICP: ', bf_icp/general_counter)
        print('ADDs of '+class_name+' after ICP: ', af_icp/general_counter)
        print ('Mean Offset: ', np.mean(offsets))
        # print('AUC of ' + class_name + ' before ICP: ', metrics.auc(auc_threshold, class_auc_adds_count[0]/general_counter)/0.1)
        # print('AUC of ' + class_name + ' after ICP: ', metrics.auc(auc_threshold, class_auc_adds_count[1]/general_counter)/0.1)

        with open(output_dir + 'results.txt', 'a') as f:
            f.write('Gaussian Mean: '+str(mean_radius_mm)+'\n')
            f.write('Gaussian Std Dev: '+str(std_dev_mm)+'\n')
            f.write('ADDs of '+class_name+' before ICP: '+str(bf_icp/general_counter)+'\n')
            f.write('ADDs of '+class_name+' after ICP: '+str(af_icp/general_counter)+'\n')
            f.write('Mean Offset: '+str(np.mean(offsets))+'\n')
            # f.write('AUC of ' + class_name + ' before ICP: '+str(metrics.auc(auc_threshold, class_auc_adds_count[0]/general_counter)/0.1)+'\n')
            # f.write('AUC of ' + class_name + ' after ICP: '+str(metrics.auc(auc_threshold, class_auc_adds_count[1]/general_counter)/0.1)+'\n')
            f.write('='*20+'\n')

    # return np.mean(offsets), bf_icp/general_counter, af_icp/general_counter, metrics.auc(auc_threshold, class_auc_adds_count[0]/general_counter)/0.1, metrics.auc(auc_threshold, class_auc_adds_count[1]/general_counter)/0.1
    return np.mean(offsets), bf_icp/general_counter, af_icp/general_counter







if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # ../datasets/test/  , D:/
    parser.add_argument('--root_dataset',
                    type=str,
                    default='../datasets/')



    opts = parser.parse_args()

    output_dir = 'logs/keypoint_test/6/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    means = []
    stds = []
    bf_icps = []
    af_icps = []
    mean_offsets = []
    i=0

    while i < 1.0:
        mean = i
        std = 0
        means.append(mean)
        stds.append(std)
        mean_offset, bf_icp, af_icp  = estimate_6d_pose_lm(opts, mean, std)

        mean_offsets.append(mean_offset)
        bf_icps.append(bf_icp)
        af_icps.append(af_icp)

        plt.plot(mean_offsets, bf_icps, label='Before ICP')
        plt.plot(mean_offsets, af_icps, label='After ICP')

        plt.xlabel('Mean Offset (mm)')
        plt.ylabel('ADDs')
        plt.legend()
        plt.savefig(output_dir + 'ADDs_vs_MeanOffset.png')
        plt.close()      

        plt.plot(means, bf_icps, label='Before ICP')
        plt.plot(means, af_icps, label='After ICP')
        plt.xlabel('Gaussian Offset (mm)')
        plt.ylabel('ADDs')
        plt.legend()
        plt.savefig(output_dir + 'ADDs_vs_GaussianMean.png')
        plt.close()

        plt.plot(stds, bf_icps, label='Before ICP')
        plt.plot(stds, af_icps, label='After ICP')
        plt.xlabel('Gaussian Std Dev (mm)')
        plt.ylabel('ADDs')
        plt.legend()
        plt.savefig(output_dir + 'ADDs_vs_StdDev.png')
        plt.close()
        
        i+=0.01
