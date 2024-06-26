import os
import argparse
import torch
import yaml

def get_log_dir(model_name, cfg):
    name = model_name
    log_dir = os.path.join('logs', name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    return log_dir


def get_config():
    return {
        1:
        dict(
            max_iteration=700000,
            lr=1e-4,
            momentum=0.99,
            betas = (0.9,0.999),
            weight_decay=0,
            interval_validate=1000,
        )
    }

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_checkpoint(model, optimizer, filename='model_best.pth.tar'):
    start_epoch = 0
    loss = []
    if os.path.isfile(filename):
        #print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        loss = checkpoint['loss']
        #print("=> loaded checkpoint '{}'"
        #          .format(filename))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, start_epoch, optimizer, loss