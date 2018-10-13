import time
import collections
import pickle
import os

import numpy as np
from datetime import datetime
from common.logger import Logger
import matplotlib
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

font = {'family' : 'monospace',
        'size'   : 24}
matplotlib.rc('font', **font)

def smooth(y,box_size):
    y_hat=np.zeros(y.shape,dtype=y.dtype)
    for i in range(y.size):
        if i < box_size//2:
            y_hat[i]=np.mean(y[:i+box_size//2])
        elif i<y.size-box_size//2:
            y_hat[i]=np.mean(y[i-box_size//2:i+box_size//2])
        else:
            y_hat[i]=np.mean(y[i-box_size//2:])
    return y_hat

keys_ = [ 'No', 'OptiName', 'op', 'kn', 'lr', 'beta1', 'beta2', ]
def get_name(dir_dict):
    if dir_dict['OptiName'].startswith('adashift'):
        return 'AdaShift lr:%s No:%s'%(dir_dict['lr'],dir_dict['No'])
    elif dir_dict['OptiName'].startswith('adam'):
        return 'Adam lr:%s No:%s'%(dir_dict['lr'],dir_dict['No'])
    elif dir_dict['OptiName'].startswith('amsgrad'):
        return 'AMSGrad lr:%s No:%s'%(dir_dict['lr'], dir_dict['No'])
    else:
        raise NameError('No Matching')



if __name__=='__main__':
    result_parent = './result/'
    result_child = '/cifar10_ini0.01_wgans_maxgp0.1_relu_max0.9_0.99_bs64_lr0.0001_0.0001_128fres2k3act0upconv_128fres2k3act0downconv_sphz100/'
    result_dirs=os.listdir(result_parent)

    Data = {}
    for result_dir in sorted(result_dirs):
        dir_splits = result_dir.split('_')
        dir_dict = dict(zip(keys_, dir_splits[0:7]))

        logger=Logger()
        test_dir=result_parent+result_dir+result_child
        logger.set_dir(test_dir)
        x_vals,y_vals=logger.load_and_return()

        y_vals=np.array(y_vals)
        name = get_name(dir_dict)
        Data[dir_splits[0]] = { 'info':dir_dict,'name':name, 'y_vals':y_vals, 'x_vals':x_vals }
                
    plt_start,plt_stop = 0 , 120000

    # this list is used to decide the plot targets and their order
    ordered_keys = sorted(Data.keys())
    
    figureNo=1
    figsize = (8,6)
    legend_prop = {'size': 24}
    smooth_size=1000
    labels=[]
    plt.close('all')
    plt.figure(figureNo,figsize=figsize)
    for Nokey in ordered_keys:
        y_vals =  Data[Nokey]['y_vals'][plt_start:plt_stop]
        plt.plot(smooth(np.array(y_vals),smooth_size))
        labels.append(Data[Nokey]['name'].split(' ')[0])

    plt.legend(labels, loc = 'upper right',prop=legend_prop)
    plt.xlabel('iteration')
    plt.ylabel('training loss')
    plt.tight_layout()

