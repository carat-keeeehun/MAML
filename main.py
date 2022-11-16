import os
import sys
import pickle
import argparse
import yaml
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "4" 

import torch
import numpy as np
from MiniImagenet import MiniImagenet
import scipy.stats
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from meta import Meta, ProtoMeta

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    
    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    new_config.device = device

    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    
    return args, new_config

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def main():
    args, config = parse_args_and_config()
    
    if config.training.larger_backbone:
        '''
        Backbone model architecture follows the Finn`17, which consists of 4 modules with 3x3 convolution layers.
        Each module contains ReLU nolinearity, batch normalization and 2x2 max pooling.
        For MiniImagenet, we use 32 filters per layer.
        '''
        # larger backbone model containes 6 modules (Task 2)
        model_arch = [
            ('conv2d', [32, 3, 3, 3, 1, 0]),
            ('relu', [True]),
            ('bn', [32]),
            ('max_pool2d', [2, 1, 0]),
            
            ('conv2d', [32, 32, 3, 3, 1, 0]),
            ('relu', [True]),
            ('bn', [32]),
            ('max_pool2d', [2, 1, 0]),
            
            ('conv2d', [32, 32, 3, 3, 1, 0]),
            ('relu', [True]),
            ('bn', [32]),
            ('max_pool2d', [2, 2, 0]),
            
            ('conv2d', [32, 32, 3, 3, 1, 0]),
            ('relu', [True]),
            ('bn', [32]),
            ('max_pool2d', [2, 2, 0]),
            
            ('conv2d', [32, 32, 3, 3, 1, 0]),
            ('relu', [True]),
            ('bn', [32]),
            ('max_pool2d', [2, 2, 0]),
            
            ('conv2d', [32, 32, 3, 3, 1, 0]),
            ('relu', [True]),
            ('bn', [32]),
            ('max_pool2d', [2, 1, 0]),
            
            ('flatten', []),
            ('linear', [config.training.n_way, 32 * 5 * 5])
        ]
    else:
        # backbone model of Finn`17 which containes 4 modules (Task 1)
        model_arch = [
            ('conv2d', [32, 3, 3, 3, 1, 0]),
            ('relu', [True]),
            ('bn', [32]),
            ('max_pool2d', [2, 2, 0]),
            
            ('conv2d', [32, 32, 3, 3, 1, 0]),
            ('relu', [True]),
            ('bn', [32]),
            ('max_pool2d', [2, 2, 0]),
            
            ('conv2d', [32, 32, 3, 3, 1, 0]),
            ('relu', [True]),
            ('bn', [32]),
            ('max_pool2d', [2, 2, 0]),
            
            ('conv2d', [32, 32, 3, 3, 1, 0]),
            ('relu', [True]),
            ('bn', [32]),
            ('max_pool2d', [2, 1, 0]),
            
            ('flatten', []),
            ('linear', [config.training.n_way, 32 * 5 * 5])
        ]
    
    '''
    We borrow model and learning part from https://github.com/dragen1860/MAML-Pytorch.
    We utilize open-source codes, and episodic learning with inner/outer optimization loop is in Meta class.
    '''
    device = config.device

    # Dataset split(MiniImagenet)
    mini = MiniImagenet('miniimagenet/', mode='train',
                        n_way=config.training.n_way,
                        k_shot=config.training.k_shot,
                        k_query=config.training.k_qry,
                        batchsz=10000,
                        resize=config.data.image_size)
    mini_val = MiniImagenet('miniimagenet/', mode='val',
                        n_way=config.training.n_way,
                        k_shot=config.training.k_shot,
                        k_query=config.training.k_qry,
                        batchsz=1000,
                        resize=config.data.image_size)
    mini_test = MiniImagenet('miniimagenet/', mode='test',
                             n_way=config.training.n_way,
                             k_shot=config.training.k_shot,
                             k_query=config.training.k_qry,
                             batchsz=100,
                             resize=config.data.image_size)

    '''
    We borrow prototype model and learning part from https://github.com/jaeho3690/Pytorch_Proto-MAML_implementation.
    We utilize open-source codes, and reorganize them suitable for our base code.
    '''
    if config.training.prototype:
        # prototype-maml (Task 3)
        protomaml = ProtoMeta(config).to(device)

        for epoch in range(config.training.epoch):
            # fetch meta_batchsz num of episode each time
            db = DataLoader(mini, config.training.meta_batch_size, shuffle=True, num_workers=1, pin_memory=True)

            for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):

                x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

                accs = protomaml(x_spt, y_spt, x_qry, y_qry)

                if step % 100 == 0:
                    print('step:', step, '\ttraining acc:', accs)
                
                if step % 5000 == 0 and step != 0:
                    # validation
                    db_val = DataLoader(mini_val, 1, shuffle=True, num_workers=1, pin_memory=True)
                    accs_all_val = []
                    
                    for x_spt, y_spt, x_qry, y_qry in db_val:
                        x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), \
                                                    x_qry.to(device), y_qry.to(device)
                        
                        accs = protomaml.validation(x_spt, y_spt, x_qry, y_qry)
                        accs_all_val.append(accs)
                    
                    accs = np.array(accs_all_val).mean(axis=0).astype(np.float16)
                    print('Validation acc:', accs)   

                if step % 500 == 0:
                    # test
                    db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
                    accs_all_test = []

                    for x_spt, y_spt, x_qry, y_qry in db_test:
                        x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), \
                                                    x_qry.to(device), y_qry.to(device)

                        accs = protomaml.test(x_spt, y_spt, x_qry, y_qry)
                        accs_all_test.append(accs)

                    # [b, test_update_steps + 1]
                    accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                    print('Test acc:', accs)        
        
    else:
        maml = Meta(config, model_arch).to(device)
    
        for epoch in range(config.training.epoch):
            # fetch meta_batchsz num of episode each time
            db = DataLoader(mini, config.training.meta_batch_size, shuffle=True, num_workers=1, pin_memory=True)

            for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):

                x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

                accs = maml(x_spt, y_spt, x_qry, y_qry)

                if step % 100 == 0:
                    print('step:', step, '\ttraining acc:', accs)

                if step % 500 == 0:  # evaluation
                    db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
                    accs_all_test = []

                    for x_spt, y_spt, x_qry, y_qry in db_test:
                        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                    x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                        accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                        accs_all_test.append(accs)

                    # [b, test_update_steps + 1]
                    accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                    print('Test acc:', accs)

if __name__ == "__main__":
    sys.exit(main())