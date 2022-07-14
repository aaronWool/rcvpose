import torch
from utils import get_config, get_log_dir, str2bool
from data_loader import get_loader
from train import Trainer
import warnings
from tensorboardX import SummaryWriter
warnings.filterwarnings('ignore')

#from visualizer import Visualizer

resume = ''

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # Parameters to set
    parser.add_argument('--mode',
                        type=str,
                        default='train',
                        choices=['train', 'test'])
    parser.add_argument("--gpu_id", type=int, default=-1)
    parser.add_argument("--root_dataset",
                        type=str,
                        default='./datasets/LINEMOD')
    parser.add_argument("--resume_train", type=str2bool, default=False)
    parser.add_argument("--optim",
                        type=str,
                        default='Adam',
                        choices=['Adam', 'SGD'])
    parser.add_argument("--batch_size",
                        type=str,
                        default='4')
    parser.add_argument("--class_name",
                        type=str,
                        default='ape')
    parser.add_argument("--initial_lr",
                        type=float,
                        default=1e-4)
    parser.add_argument("--kpt_num",
                        type=str,
                        default='1')                        
    parser.add_argument('--model_dir',
                    type=str,
                    default='ckpts/')   
    parser.add_argument('--demo_mode',
                    type=bool,
                    default=False) 
    parser.add_argument('--test_occ',
                    type=bool,
                    default=False) 
    opts = parser.parse_args()

    cfg = get_config()[1]
    opts.cfg = cfg

    if opts.mode in ['train']:
        opts.out = get_log_dir(opts.class_name+'Kp'+opts.kpt_num, cfg)
        print('Output logs: ', opts.out)
        vis = SummaryWriter(logdir=opts.out+'/tbLog/')
    else:
        vis = []

    data = get_loader(opts)
    
    trainer = Trainer(data, opts, vis)
    if opts.mode == 'test':
        trainer.Test()
    else:
        trainer.Train()
