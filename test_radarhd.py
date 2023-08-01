#  File for testing RadarHD

import time
import os
import datetime
import json
from functools import partial

import torch
from scipy.optimize import brentq

import numpy as np
# from torchsummary import summary

from train_test_utils.dataloader import *
from train_test_utils.model import *

"""
## Constants. Edit this to change the model to test on.
"""

params = {
    'model_name': '13',
    'expt': 1,
    'dt': '20220320-034822',
    'epoch_num': 120,
    'data': 5,
    'gpu': 1,
}

def dataloader(train_params):
    print('Loading data')
    basepath = './dataset_' + str(params['data']) + '/'

    orig_size = [256, 64, 512]
    reqd_size = [256, 64, 512]

    test_set = Dataset(basepath, 'test',
                        RBINS=reqd_size[0], ABINS_RADAR=reqd_size[1], ABINS_LIDAR=reqd_size[2],
                        RBINS_ORIG=orig_size[0], ABINS_RADAR_ORIG=orig_size[1], ABINS_LIDAR_ORIG=orig_size[2], 
                        M=train_params['history'])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    ordered_filename = test_set.__filenames__()
    print('# of points to test: ', len(test_loader))
    return (test_loader, ordered_filename)

def val_dataloader(train_params):
    print('Loading data')
    basepath = './dataset_' + str(params['data']) + '/'

    orig_size = [256, 64, 512]
    reqd_size = [256, 64, 512]

    val_set = Dataset(basepath, 'val',
                        RBINS=reqd_size[0], ABINS_RADAR=reqd_size[1], ABINS_LIDAR=reqd_size[2],
                        RBINS_ORIG=orig_size[0], ABINS_RADAR_ORIG=orig_size[1], ABINS_LIDAR_ORIG=orig_size[2], 
                        M=train_params['history'])
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=len(val_set), shuffle=False)

    ordered_filename = val_loader.__filenames__()
    print('# of points to test: ', len(val_loader))
    return (val_loader, ordered_filename)

def main():
    print(torch.__version__)
    torch.manual_seed(0)  

    # Can be set to cuda/cpu. Make sure model and data are moved to cuda if cuda is used
    if params['gpu'] == 1:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    def false_positive_rate(pred_masks, false_masks):
        return ((pred_masks * false_masks).sum(axis=1).sum(axis=1)/false_masks.sum(axis=1).sum(axis=1)).mean()
    
    def lamhat_threshold(cal_sgmd, cal_gt_masks, n, alpha, lam): 
        return false_positive_rate(cal_sgmd>=lam, cal_gt_masks) - ((n+1)/n*alpha - 1/(n+1))

    name_str = params['model_name'] + '_' + str(params['expt']) + '_' + params['dt']
    LOG_DIR = './logs/' + name_str + '/'
    with open(os.path.join(LOG_DIR, 'params.json'), 'r') as f:
        train_params = json.load(f)

    # Load data
    (test_loader, ordered_filename) = dataloader(train_params)
    (val_loader, val_ordered_filename) = val_dataloader(train_params)

    # Define model
    gen = UNet1(train_params['history']+1, 1).to(device)
    # summary(gen, (train_params['history']+1, 256, 64))

    epoch_num = '%03d' % params['epoch_num']
    model_file = LOG_DIR + epoch_num + '.pt_gen'
    checkpoint = torch.load(model_file, map_location=device)
    gen.load_state_dict(checkpoint['state_dict'])

    save_path = './logs/' + name_str + '/test_imgs/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Testing
    gen.eval()
    
    # Split the softmax scores into calibration and validation sets (save the shuffling)
    val_data, val_label = val_loader.__next__()
    val_data, val_label = val_data.to(device), val_label.to(device)
    # Run the conformal risk control procedure
    with torch.no_grad():
        val_pred_score = gen(val_data)
        lamhat_threshold_partial = partial(lamhat_threshold, val_pred_score, val_label, len(val_loader), 0.1)
        lamhat = brentq(lamhat_threshold_partial, 0, 1)
        val_pred = val_pred_score >= lamhat
        # Calculate empirical FNR
        print(f"The empirical FNR is: {false_positive_rate(val_pred, val_label)} and the threshold value is: {lamhat}")
        
    t0 = time.time()
    for test_i, (test_data, test_label) in enumerate(test_loader):

        test_data, test_label = test_data.to(device), test_label.to(device)
        with torch.no_grad():
            pred = gen(test_data)
            
            pred = np.squeeze(pred.cpu().numpy())
            # pred = (pred*255).astype(np.uint8)
            # for conformal risk control
            pred = pred >= lamhat
            im1 = Image.fromarray(pred)

            im1_file_name = save_path + epoch_num + '_' + ordered_filename[test_i] + '_pred.png'
            im1.save(im1_file_name)
            
            label = np.squeeze(test_label.cpu().numpy())
            label = (label*255).astype(np.uint8)
            im1 = Image.fromarray(label)
            im1_file_name = save_path + epoch_num + '_' + ordered_filename[test_i] + '_label.png'
            im1.save(im1_file_name)
            
            print(ordered_filename[test_i])

    t1 = time.time()
    print('Time taken for inference: ' ,t1 - t0)

main()