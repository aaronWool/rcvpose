from util.horn import HornPoseFitting
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import open3d as o3d
import time
from PIL import Image
import math
#import h5py
from sklearn import metrics
from boptoolkit.pose_error import vsd, mssd, mspd, add, adi, re, te, proj


# lm_cls_names = ['ape', 'benchvise', 'cam', 'can', 'cat', 'duck', 'driller', 'eggbox', 'glue', 'holepuncher','iron','lamp','phone']

lm_cls_names = ['ape']

lm_syms = ['eggbox', 'glue']

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



depthList=[]

def estimate_6d_pose_lm(opts, mean_radius_mm, std_dev_mm):
    horn = HornPoseFitting()

    offsets = []
    bf_icp_add = []
    af_icp_add = []

    bf_icp_adi = []
    af_icp_adi = []

    bf_icp_re = []
    af_icp_re = []

    bf_icp_te = []
    af_icp_te = []

    test_o3d_icp_scores = []

    bf_icp_proj = []
    af_icp_proj = []


    for class_name in lm_cls_names:


        # print("Evaluation on ", class_name)
        rootPath = opts.root_dataset + "LINEMOD_ORIG/"+class_name+"/"
        # rootpvPath = opts.root_dataset +class_name+"/"
        rootpvPath = opts.root_dataset +"LINEMOD/"+class_name+"/"
        # test_list = open(opts.root_dataset +class_name+"/" +"Split/val.txt","r").readlines()
        test_list = open(opts.root_dataset +"LINEMOD/"+ class_name+"/" +"Split/train.txt","r").readlines()
        test_list = [ s.replace('\n', '') for s in test_list]
        test_list_len = len(test_list)
        

        # pcd_load = o3d.io.read_point_cloud(opts.root_dataset +class_name+"/"+class_name+".ply")
        pcd_load = o3d.io.read_point_cloud(opts.root_dataset +"LINEMOD/"+class_name+"/"+class_name+".ply")

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
        # for filename in os.listdir(dataPath):
            input_path = dataPath + filename
            # RTGT = np.load(opts.root_dataset +class_name+"/pose/pose"+str(int(os.path.splitext(filename)[0]))+'.npy')
            RTGT = np.load(opts.root_dataset +"LINEMOD/"+class_name+"/pose/pose"+str(int(os.path.splitext(filename)[0]))+'.npy')
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
                        # z = 0
                        z = random_radius*np.cos(phi)
                        

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
                    # print(RTGT)
                    # print(RT)

                    dump, xyz_load_est_transformed=project(xyz_load*1000, linemod_K, RT[0:3,:])
                    RTGT_mm = RTGT
                    RTGT_mm[:,3] = RTGT_mm[:,3]*1000

                    # print(RTGT_mm)
                    dump, xyz_load_transformed=project(xyz_load*1000, linemod_K, RTGT_mm)

                    R_est = RT[0:3,0:3]
                    R_gt = RTGT_mm[0:3,0:3]
                    t_est = RT[0:3,3]
                    t_gt = RTGT_mm[0:3,3]
      
                    add_error = add(R_est, t_est, R_gt, t_gt, xyz_load)
                    adi_error = adi(R_est, t_est, R_gt, t_gt, xyz_load)
                    re_error = re(R_est, R_gt)
                    te_error = te(t_est, t_gt)
                    # proj_error = proj(R_est, t_est, R_gt, t_gt, linemod_K, xyz_load)

                    bf_icp_add.append(add_error)
                    bf_icp_adi.append(adi_error)
                    bf_icp_re.append(re_error)
                    bf_icp_te.append(te_error)
                    # bf_icp_proj.append(proj_error)

                    if add_error <= add_threshold[class_name]*1000:
                        bf_icp+=1

                    # trans_init = np.eye(4)
                    
                    trans_init = RT

                    threshold = add_error

                    sceneGT = o3d.geometry.PointCloud()
                    sceneEst = o3d.geometry.PointCloud()

                    sceneGT.points = o3d.utility.Vector3dVector(xyz_load_transformed)
                    sceneEst.points = o3d.utility.Vector3dVector(xyz_load_est_transformed)

                    sceneGT.paint_uniform_color(np.array([0, 0, 1]))
                    sceneEst.paint_uniform_color(np.array([1, 0, 0]))

                    cad_model = o3d.geometry.PointCloud()
                    cad_model.points = o3d.utility.Vector3dVector(xyz_load*1000)

                    
                    criteria = o3d.pipelines.registration.ICPConvergenceCriteria()
                    reg_p2p = o3d.pipelines.registration.registration_icp(
                        cad_model, sceneEst,  threshold, trans_init,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                        criteria)
                    
                    cad_model.transform(reg_p2p.transformation)
                    
                    icp_RT = np.asarray(reg_p2p.transformation) 

                    mean_dist = np.asarray(sceneGT.compute_point_cloud_distance(cad_model)).mean()
                    
                    
                    R_est_icp = icp_RT[0:3,0:3]
                    t_est_icp = icp_RT[0:3,3]

                    add_error_icp = add(R_est_icp, t_est_icp, R_gt, t_gt, xyz_load)
                    adi_error_icp = adi(R_est_icp, t_est_icp, R_gt, t_gt, xyz_load)
                    re_error_icp = re(R_est_icp, R_gt)
                    te_error_icp = te(t_est_icp, t_gt)
                    # proj_error_icp = proj(R_est_icp, t_est_icp, R_gt, t_gt, linemod_K, xyz_load)

                    af_icp_add.append(add_error_icp)
                    af_icp_adi.append(adi_error_icp)
                    af_icp_re.append(re_error_icp)
                    af_icp_te.append(te_error_icp)
                    test_o3d_icp_scores.append(mean_dist)
                    # af_icp_proj.append(proj_error_icp)
                               

                    if add_error_icp <= add_threshold[class_name]*1000:
                        af_icp+=1

                    general_counter += 1

                    if general_counter/test_list_len == 1:
                        break

                        

                    # print (xyz_load_transformed)
                    # print (xyz_load_est_transformed)
                    # exit()
                    # xyz_load_transformed = xyz_load_transformed / 1000
                    # xyz_load_est_transformed = xyz_load_est_transformed / 1000
                    
                    # xyz_load_est_transformed = xyz_load_est_transformed*1000
     
                    # input_image = np.asarray(Image.open(input_path).convert('RGB'))
                    # input_image = np.copy(input_image)
                    # for coor in dump:
                        # if coor[0] >= 0 and coor[0] < input_image.shape[1] and coor[1] >= 0 and coor[1] < input_image.shape[0]:
                            # input_image[int(coor[1]),int(coor[0])] = [255,0,0]
                    # plt.imshow(input_image)
                    # plt.show()

            
                    # sceneGT = o3d.geometry.PointCloud()
                    # sceneEst = o3d.geometry.PointCloud()
                    # sceneGT.points = o3d.utility.Vector3dVector(xyz_load_transformed)
                    # sceneEst.points = o3d.utility.Vector3dVector(xyz_load_est_transformed)
                    # sceneGT.paint_uniform_color(np.array([0, 0, 1]))
                    # sceneEst.paint_uniform_color(np.array([1, 0, 0]))

                    # min_distance = np.asarray(sceneGT.compute_point_cloud_distance(sceneEst)).min()
                    # distance = np.asarray(sceneGT.compute_point_cloud_distance(sceneEst)).mean()


                    # if class_name in lm_syms:
                    #     bf_icp_dist.append(min_distance)
                    #     if min_distance <= add_threshold[class_name]*1000:
                    #         bf_icp+=1
                    # else:
                    #     bf_icp_dist.append(distance)
                    #     #print('ADD(s) point distance before ICP: ', distance)
                    #     if distance <= add_threshold[class_name]*1000:
                    #         bf_icp+=1

                    # i = 0
                    # for threshold in auc_threshold:
                    #     if class_name in lm_syms:
                    #         if min_distance <= threshold*1000:
                    #             auc_adds_count[0, i] += 1
                    #             class_auc_adds_count [0, i] += 1
                    #     else:
                    #         if distance <= threshold*1000:
                    #             auc_adds_count[0, i] += 1
                    #             class_auc_adds_count [0, i] += 1
                    #     i += 1

                    # # print ('Distance before ICP: ', distance)
                    # # print ('Successful before ICP: ', bf_icp)

                    # # o3d.visualization.draw_geometries([sceneGT, sceneEst],window_name='gt vs est before icp')

                    # trans_init = np.asarray([[1, 0, 0, 0],
                    #                         [0, 1, 0, 0],
                    #                         [0, 0, 1, 0], 
                    #                         [0, 0, 0, 1]])
                    # if class_name in lm_syms:
                    #     threshold = min_distance
                    # else:
                    #     threshold = distance
                    # criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
                    # reg_p2p = o3d.pipelines.registration.registration_icp(
                    #     sceneGT, sceneEst,  threshold, trans_init,
                    #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    #     criteria)
                    # sceneGT.transform(reg_p2p.transformation)
             

                    # #print('ADD(s) point distance after ICP: ', distance)
                    # min_distance = np.asarray(sceneGT.compute_point_cloud_distance(sceneEst)).min()
                    # distance = np.asarray(sceneGT.compute_point_cloud_distance(sceneEst)).mean()
                    # if class_name in lm_syms:
                    #     af_icp_dist.append(min_distance)
                    #     if min_distance <= add_threshold[class_name]*1000:
                    #         af_icp+=1
                    # else:
                    #     af_icp_dist.append(distance)
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

           



        #os.system("pause")

    print('='*20)
    print ('Gaussian Mean: ', mean_radius_mm, 'mm')
    print ('Gaussian Std Dev: ', std_dev_mm, 'mm')
    print('ADDs of '+class_name+' before ICP: ', bf_icp/general_counter)
    print('ADDs of '+class_name+' after ICP: ', af_icp/general_counter)
    print ('Mean Point Cloud Distance before ICP: ', np.mean(bf_icp_add), 'mm')
    print ('Mean Point Cloud Distance after ICP: ', np.mean(af_icp_add), 'mm')
    print ('Mean ADI before ICP: ', np.mean(bf_icp_adi), 'mm')
    print ('Mean ADI after ICP: ', np.mean(af_icp_adi), 'mm')
    print ('Mean RE before ICP: ', np.mean(bf_icp_re), 'mm')
    print ('Mean RE after ICP: ', np.mean(af_icp_re), 'mm')
    print ('Mean TE before ICP: ', np.mean(bf_icp_te), 'mm')
    print ('Mean TE after ICP: ', np.mean(af_icp_te), 'mm')
    print ('Open3d ICP Score: ', np.mean(test_o3d_icp_scores))
    print ('Mean Offset: ', np.mean(offsets))
    # print('AUC of ' + class_name + ' before ICP: ', metrics.auc(auc_threshold, class_auc_adds_count[0]/general_counter)/0.1)
    # print('AUC of ' + class_name + ' after ICP: ', metrics.auc(auc_threshold, class_auc_adds_count[1]/general_counter)/0.1)
    with open(output_dir + 'results.txt', 'a') as f:
        f.write('Gaussian Mean: '+str(mean_radius_mm)+'\n')
        f.write('Gaussian Std Dev: '+str(std_dev_mm)+'\n')
        f.write('ADDs before ICP: '+str(bf_icp/general_counter)+'\n')
        f.write('ADDs after ICP: '+str(af_icp/general_counter)+'\n')
        f.write('Mean Point Cloud Distance before ICP: '+str(np.mean(bf_icp_add))+'\n')
        f.write('Mean Point Cloud Distance after ICP: '+str(np.mean(af_icp_add))+'\n')
        f.write('Mean ADI before ICP: '+str(np.mean(bf_icp_adi))+'\n')
        f.write('Mean ADI after ICP: '+str(np.mean(af_icp_adi))+'\n')
        f.write('Mean RE before ICP: '+str(np.mean(bf_icp_re))+'\n')
        f.write('Mean RE after ICP: '+str(np.mean(af_icp_re))+'\n')
        f.write('Mean TE before ICP: '+str(np.mean(bf_icp_te))+'\n')
        f.write('Mean TE after ICP: '+str(np.mean(af_icp_te))+'\n')
        f.write('Mean Keypoint Error: '+str(np.mean(offsets))+'\n')
        # f.write('AUC of ' + class_name + ' before ICP: '+str(metrics.auc(auc_threshold, class_auc_adds_count[0]/general_counter)/0.1)+'\n')
        # f.write('AUC of ' + class_name + ' after ICP: '+str(metrics.auc(auc_threshold, class_auc_adds_count[1]/general_counter)/0.1)+'\n')
        f.write('='*20+'\n')
        
        # # histogram of offsets
        # plt.hist(offsets, bins=200, color='b', alpha=0.7, rwidth=0.85)
        # plt.xlabel('Offset (mm)')
        # plt.ylabel('Frequency')
        # plt.title('Offset Histogram')
        # if not os.path.exists(output_dir + 'Offset_Histograms/'):
        #     os.makedirs(output_dir + 'Offset_Histograms/')
        # plt.savefig(output_dir + 'Offset_Histograms/r'+str(mean_radius_mm)+'_s'+str(std_dev_mm)+'.png')
        # plt.close()

    # return np.mean(offsets), bf_icp/general_counter, af_icp/general_counter, metrics.auc(auc_threshold, class_auc_adds_count[0]/general_counter)/0.1, metrics.auc(auc_threshold, class_auc_adds_count[1]/general_counter)/0.1
    return np.mean(offsets), bf_icp/general_counter, af_icp/general_counter, np.mean(bf_icp_add), np.mean(af_icp_add), np.mean(bf_icp_adi), np.mean(af_icp_adi), np.mean(bf_icp_re), np.mean(af_icp_re), np.mean(bf_icp_te), np.mean(af_icp_te)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # ../datasets/test/  , D:/
    parser.add_argument('--root_dataset',
                    type=str,
                    default='D:/')

    opts = parser.parse_args()

    output_dir = 'logs/keypoint_test/new_metrics2/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    means = []
    stds = []
    bf_icps = []
    af_icps = []
    mean_offsets = []
    mean_dist_bf_icp = []
    mean_dist_af_icp = []
    mean_adi_bf_icp = []
    mean_adi_af_icp = []
    mean_re_bf_icp = []
    mean_re_af_icp = []
    mean_te_bf_icp = []
    mean_te_af_icp = []

    i=1

    while i < 300.0:
        mean = i
        std = i/2
        means.append(mean)
        stds.append(std)
        mean_offset, bf_icp, af_icp, dist_bf_icp, dist_af_icp, adi_bf_icp, adi_af_icp, re_bf_icp, re_af_icp, te_bf_icp, te_af_icp = estimate_6d_pose_lm(opts, mean, std)

        mean_offsets.append(mean_offset)
        bf_icps.append(bf_icp)
        af_icps.append(af_icp)
        mean_dist_bf_icp.append(dist_bf_icp)
        mean_dist_af_icp.append(dist_af_icp)
        mean_adi_bf_icp.append(adi_bf_icp)
        mean_adi_af_icp.append(adi_af_icp)
        mean_re_bf_icp.append(re_bf_icp)
        mean_re_af_icp.append(re_af_icp)
        mean_te_bf_icp.append(te_bf_icp)
        mean_te_af_icp.append(te_af_icp)

        plt.plot (means, bf_icps, 'r', label='Before ICP')
        plt.plot (means, af_icps, 'b', label='After ICP')
        plt.xlabel('Mean Radius (mm)')
        plt.ylabel('ADDs')
        plt.title('ADDs vs Mean Radius')
        plt.legend()
        plt.savefig(output_dir + 'ADDs_vs_Mean_Radius.png')
        plt.close()

        plt.plot (means, mean_dist_bf_icp, 'r', label='Before ICP')
        plt.plot (means, mean_dist_af_icp, 'b', label='After ICP')
        plt.xlabel('Mean Radius (mm)')
        plt.ylabel('Mean Point Cloud Distance (mm)')
        plt.title('Mean Point Cloud Distance vs Mean Radius')
        plt.legend()
        plt.savefig(output_dir + 'Mean_Point_Cloud_Distance_vs_Mean_Radius.png')
        plt.close()

        plt.plot (means, mean_adi_bf_icp, 'r', label='Before ICP')
        plt.plot (means, mean_adi_af_icp, 'b', label='After ICP')
        plt.xlabel('Mean Radius (mm)')
        plt.ylabel('Mean ADI (mm)')
        plt.title('Mean ADI vs Mean Radius')
        plt.legend()
        plt.savefig(output_dir + 'Mean_ADI_vs_Mean_Radius.png')
        plt.close()

        plt.plot (means, mean_re_bf_icp, 'r', label='Before ICP')
        plt.plot (means, mean_re_af_icp, 'b', label='After ICP')
        plt.xlabel('Mean Radius (mm)')
        plt.ylabel('Mean RE (mm)')
        plt.title('Mean RE vs Mean Radius')
        plt.legend()
        plt.savefig(output_dir + 'Mean_RE_vs_Mean_Radius.png')
        plt.close()

        plt.plot (means, mean_te_bf_icp, 'r', label='Before ICP')
        plt.plot (means, mean_te_af_icp, 'b', label='After ICP')
        plt.xlabel('Mean Radius (mm)')
        plt.ylabel('Mean TE (mm)')
        plt.title('Mean TE vs Mean Radius')
        plt.legend()
        plt.savefig(output_dir + 'Mean_TE_vs_Mean_Radius.png')
        plt.close()

        i+=1
