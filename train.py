import yaml
import logging
import os
import argparse
import numpy as np
import torch

from noise2noise import Noise2Noise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='input arguments to train the Unet for classification')   
    parser.add_argument('--config',  '-c',
                            dest="filename",
                            metavar='FILE',
                            help =  'path to the config file',
                            default='configs/noise2noise.yaml')
    
    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    seed = config['exp_params']['manual_seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = Noise2Noise(args, config)
    model.train()