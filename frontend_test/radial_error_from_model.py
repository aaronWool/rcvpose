import numpy as np
from PIL import Image
import matplotlib.pyplot  as plt
import os
import time
from ransac_vanilla import RANSAC_vanilla
from ransac import RANSAC_3D
from ransac_to_accumulator import RANSAC_Accumulator
import datetime
from accumulator3D import Accumulator_3D
from tqdm import tqdm
import random
import open3d as o3d
import warnings
import torch
from fcnresnet import DenseFCNResNet152
from utils import load_checkpoint
warnings.filterwarnings("ignore")

#lm_cls_names = ['ape', 'benchvise', 'cam', 'can', 'cat', 'duck', 'driller', 'eggbox', 'glue', 'holepuncher','iron','lamp','phone']

lm_cls_names = ['ape']


def FCResBackbone(model, input_img_path, normalized_depth):
    """
    This is a funciton runs through a pre-trained FCN-ResNet checkpoint
    Args:
        model: model obj
        input_img_path: input image to the model
    Returns:
        output_map: feature map estimated by the model
                    radial map output shape: (1,h,w)
                    vector map output shape: (2,h,w)
    """
    #model = DenseFCNResNet152(3,2)
    #model = torch.nn.DataParallel(model)
    #checkpoint = torch.load(model_path)
    #model.load_state_dict(checkpoint)
    #optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    #model, _, _, _ = utils.load_checkpoint(model, optim, model_path)
    #model.eval()
    input_image = Image.open(input_img_path).convert('RGB')
    #plt.imshow(input_image)
    #plt.show()
    img = np.array(input_image, dtype=np.float64)
    img /= 255.
    img -= np.array([0.485, 0.456, 0.406])
    img /= np.array([0.229, 0.224, 0.225])
    img = img.transpose(2, 0, 1)
    #dpt = np.load(normalized_depth)
    #img = np.append(img,np.expand_dims(dpt,axis=0),axis=0)
    input_tensor = torch.from_numpy(img).float()

    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    # use gpu if available
    if torch.cuda.is_available():
         input_batch = input_batch.to('cuda')
         model.to('cuda')
    with torch.no_grad():
        sem_out, radial_out = model(input_batch)
    sem_out, radial_out = sem_out.cpu(), radial_out.cpu()

    sem_out, radial_out = np.asarray(sem_out[0]),np.asarray(radial_out[0])
    return sem_out[0], radial_out[0]



linemod_K = np.array([[572.4114, 0., 325.2611],
                  [0., 573.57043, 242.04899],
                  [0., 0., 1.]])

def rgbd_to_point_cloud(K, depth):
    vs, us = depth.nonzero()
    zs = depth[vs, us]
    #print(zs.min())
    #print(zs.max())
    xs = ((us - K[0, 2]) * zs) / float(K[0, 0])
    ys = ((vs - K[1, 2]) * zs) / float(K[1, 1])
    pts = np.array([xs, ys, zs]).T
    return pts

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


def test_epsilon(root_dataset, out_dir):
    epsilons = []
    
    for class_name in lm_cls_names:
        print ('class_name:', class_name)

        model_list = []


        class_epsilons = []

        if not os.path.exists(out_dir + class_name):
            os.makedirs(out_dir + class_name)
        

        # Root paths for dataset and file list
        rootPath = root_dataset + "LINEMOD_ORIG/"+class_name+"/" 
        rootpvPath = root_dataset + "LINEMOD/"+class_name+"/" 
        img_path = root_dataset + "LINEMOD/"+class_name+"/" + "JPEGImages/"
        # rootRadialMapPath = root_dataset + "rkhs_estRadialMap/"+class_name+"/"
        # rootRadialMapPath = root_dataset + "LINEMOD/"+class_name+"/"
        rootRadialMapPath = root_dataset + "estRadialMap/"+class_name+"/"
        test_list = open(root_dataset + "LINEMOD/"+class_name+"/" +"Split/val.txt","r").readlines()
        test_list = [ s.replace('\n', '') for s in test_list]
        test_list_size = len(test_list)

        model_path = rootpvPath + 'models/'
        models = os.listdir(model_path)
        print ('models:', models)
        for model in models:
            tmp = DenseFCNResNet152(3,2)
            optim = torch.optim.Adam(tmp.parameters(), lr=1e-3)
            tmp, _, _, _ = load_checkpoint(tmp, optim, model_path + model)
            tmp.eval()
            model_list.append(tmp)

        # Load object pointcloud
        pcd_load = o3d.io.read_point_cloud(root_dataset + "LINEMOD/"+class_name+"/"+class_name+".ply")
        xyz_load = np.asarray(pcd_load.points)

        # Load KeyGNet keypoints
        # keypoints=np.load(root_dataset + "rkhs_estRadialMap/KeyGNet_kpts 1.npy")
        # keypoints = keypoints[0:3]
        # keypoints = keypoints / 1000

        # load outside 9
        keypoints=np.load(rootpvPath + 'Outside9.npy')

        max_radii_dm = np.zeros(3)
        for i in range(3):
            dsitances = ((xyz_load[:,0]-keypoints[i+1,0])**2
                 +(xyz_load[:,1]-keypoints[i+1,1])**2
                +(xyz_load[:,2]-keypoints[i+1,2])**2)**0.5
            max_radii_dm[i] = dsitances.max()*10


        keypoints = keypoints[1:4]
        #random_files = random.sample(test_list, 5)
        for filename in tqdm(test_list, total=len(test_list), unit='image', leave=False):
            
            # Read in rotation and translation matrix
            RTGT = np.load(root_dataset + "LINEMOD/"+class_name+"/pose/pose"+str(int(os.path.splitext(filename)[0]))+'.npy')
            
            # transform keypoints to GT pose and mm scale
            kpGT_mm = (np.dot(keypoints, RTGT[:, :3].T) + RTGT[:, 3:].T)*1000


            keypoint_count = 0
            for keypoint in keypoints:

                centerGT_mm = kpGT_mm[keypoint_count]

                # load image
                img = Image.open(img_path + filename + '.jpg')
                img = np.array(img)

                # pass image through FCResNet

                semMap, radMap = FCResBackbone(model_list[keypoint_count], img_path + filename + '.jpg', rootRadialMapPath + filename + '.npy')
                semMap = np.where(radMap<=max_radii_dm[keypoint_count], semMap, 0)
                radMap = np.where(radMap<=max_radii_dm[keypoint_count], radMap, 0)
                semMap = np.where(semMap>0.8, 1, 0)

                # load depth map and mask it with semMask
                depthMap = read_depth(rootPath+'data/depth'+str(int(os.path.splitext(filename)[0]))+'.dpt')
                maskedDepthMap = depthMap*semMap
                
                # apply the same mask to radMap
                maskedRadMap = radMap*semMap
                
                # get pixel coordinates of the mask
                pixelCoords = np.where(maskedDepthMap>0) 
                
                # get radial values of the mask
                radList = maskedRadMap[pixelCoords]
                radList = radList*100

                # get 3D coordinates of the mask
                xyz_mm = rgbd_to_point_cloud(linemod_K, maskedDepthMap)

                if xyz_mm.shape[0] != radList.shape[0]:
                    print ('xyz_mm:', xyz_mm.shape)
                    print ('radList:', radList.shape)
                    print ('filename:', filename)
                    print ('keypoint_count:', keypoint_count)


                assert xyz_mm.shape[0] == radList.shape[0], "Number of points in depth map and radial map do not match"
                assert xyz_mm.shape[0] != 0, "No points found in depth map"
             

                for xyz, rad in zip(xyz_mm, radList):
                    # compute offset between GT and KeyGNet keypoints
                    offset = np.linalg.norm(xyz - centerGT_mm)

                    # compute epsilon in mm
                    current_epsilon = abs(offset - rad)

                    epsilons.append(current_epsilon)
                    class_epsilons.append(current_epsilon)

                keypoint_count += 1
                if keypoint_count == 3:
                    break

  
            #plt.hist(epsilons, bins=300)
            plt.title('Epsilon Histogram')
            plt.xlabel('Epsilon')
            plt.ylabel('Frequency')
            plt.savefig(out_dir + class_name + '/' + str(filename) + '.png')
            plt.close() 
        

      
        print ('Radial Error Mean[mm]: ', np.mean(class_epsilons))
        print ('Radial Error Std[mm]: ', np.std(class_epsilons))
        plt.hist(class_epsilons, bins=2000)
        plt.title(class_name + ' Epsilon Histogram')
        plt.xlabel('Epsilon')
        plt.ylabel('Frequency')
        plt.savefig(out_dir + class_name + '/epsilon_hist.png')
        #plt.show()
        plt.close()




if __name__ == "__main__":

    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dataset',
                        type=str,
                        default='D:/')
    
    out_dir = 'logs/epsilon_test2/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    args = parser.parse_args()

    root_dataset = args.root_dataset

    print ('root_dataset:', args.root_dataset)
    print ('out_dir:', out_dir)
    test_epsilon(root_dataset, out_dir)
