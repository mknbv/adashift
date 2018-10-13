import numpy as np 
import os
from matplotlib import pyplot as plt
from math import sqrt,exp
import math
import time
from random import random
from scipy.signal import savgol_filter
import matplotlib

color_map={
        'SGD':                              '#BBBB00',
        'Adam beta1:0.0':                   '#0066FF',     
        'Adam beta1:0.9':                   '#003C9D',  
        'AMSGrad beta1:0.0':                '#00DD00',
        'AMSGrad beta1:0.9':                '#008844',
        'AdaShift op:max n:1 beta1:0.0':    '#FF3333',
        'max-AdaShift op:max n:1 beta1:0.0':    '#FF3333',
        'AdaShift op:max n:10 beta1:0.9':   '#CC0000',
        'max-AdaShift op:max n:10 beta1:0.9':   '#CC0000',
        'AdaShift op:max n:10 beta1:1.0':   '#AA0000',
        'AdaShift op:none n:1 beta1:0.0':   '#EE7700',
        'non-AdaShift op:none n:1 beta1:0.0':   '#EE7700',
        'AdaShift op:none n:10 beta1:0.9':  '#BB5500',
        'non-AdaShift op:none n:10 beta1:0.9':  '#BB5500',
        'AdaShift op:none n:10 beta1:1.0':  '#A42D00',
    }
def color_map_fun(key,default_color = '#99FF33'):
    color = color_map[key] if key in color_map.keys() else default_color
    return color
    


font = {'family' : 'monospace',
        'size'   : 24}
matplotlib.rc('font', **font)

def smooth(y,box_size,smooth_start=0):
    y_hat=np.zeros(y.shape,dtype=y.dtype)
    y_hat[0:smooth_start]=y[0:smooth_start]
    for i in range(smooth_start,y.size):
        if i < smooth_start+box_size//2:
            y_hat[i]=np.mean(y[smooth_start:i+box_size//2])
        elif i<y.size-box_size//2:
            y_hat[i]=np.mean(y[i-box_size//2:i+box_size//2])
        else:
            y_hat[i]=np.mean(y[i-box_size//2:])
    return y_hat

def get_label(target_list):
    if target_list[1].startswith('adashift'):
        label="AdaShift op:%s n:%s beta1:%s.%s"%( target_list[2], target_list[3], target_list[5][0], target_list[5][2] )
    elif target_list[1].startswith('adamshift'):
        label="%s-AdaShift op:%s n:1 beta1:%s.%s"%(target_list[2], target_list[2], target_list[5][0], target_list[5][2])
    elif target_list[1].startswith('adam'):
        label="Adam beta1:%s.%s"%(target_list[5][0],target_list[5][2] )
    elif target_list[1].startswith('amsgrad'):
        label="AMSGrad beta1:%s.%s"%( target_list[5][0], target_list[5][2])
    elif target_list[1].startswith('sgd'):
        label="SGD"#%( int(target_list[0]), 'SGD', float(target_list[4]) )
    else:
        label="%d %s op:%s kn:%s lr:%.3f beta1:%.2f beta2:%.3f "%( int(target_list[0]), target_list[1], target_list[2], target_list[3], float(target_list[4]), float(target_list[5]),float(target_list[6]) )
    return label



log_dir='./logs/'
result_dir='/result_data/'
dirs=os.listdir(log_dir)
ifsave=0

# this list denotes the plot targets, which contains the run time ID
# You may change it to suit your own running order and run time ID
plot_targets=[
        '0', #sgd

        '1', #adam 0

        '2', #amsgrad      0.0
       
        '3', #moviing      max     1

        '4', #moving       none    1
]

color_map={
        'SGD':                              '#BBBB00',
        'Adam beta1:0.0':                   '#0066FF',     
        'Adam beta1:0.9':                   '#003C9D',  
        'AMSGrad beta1:0.0':                '#00DD00',
        'AMSGrad beta1:0.9':                '#008844',
        'AdaShift op:max n:1 beta1:0.0':    '#FF3333',
        'max-AdaShift op:max n:1 beta1:0.0':    '#FF3333',
        'AdaShift op:max n:10 beta1:0.9':   '#CC0000',
        'max-AdaShift op:max n:10 beta1:0.9':   '#CC0000',
        'AdaShift op:max n:10 beta1:1.0':   '#AA0000',
        'AdaShift op:none n:1 beta1:0.0':   '#EE7700',
        'non-AdaShift op:none n:1 beta1:0.0':   '#EE7700',
        'AdaShift op:none n:10 beta1:0.9':  '#BB5500',
        'non-AdaShift op:none n:10 beta1:0.9':  '#BB5500',
        'AdaShift op:none n:10 beta1:1.0':  '#A42D00',
    }


TrainAfile,TrainLfile,TestAfile,TestLfile='Train_Acc.npy','Train_Loss.npy','Test_Acc.npy','Test_Loss.npy'

Data={}
for target in dirs:
    target_list=target.split('_')
    handles={'Train_Acc':[],'Train_Loss':[],'Test_Acc':[],'Test_Loss':[]}
    target_resultdir=log_dir+target+result_dir
    if target_list[0] in plot_targets:
        Train_Acc=np.load(target_resultdir+TrainAfile)
        Train_Loss=np.load(target_resultdir+TrainLfile)
        Test_Acc=np.load(target_resultdir+TestAfile)
        Test_Loss=np.load(target_resultdir+TestLfile)

        if target_list[1]=='adamshif':
            print('Wrong name [adamshif]')
            target_list[1]='adamshift'
        elif target_list[1]=='adamshifN':
            print('Wrong name [adamshifN]')
            target_list[1]='adamshiftN'
        if target_list[3]=='1' and '9' in target_list[5]:
            target_list[5]='0.0'
        name = get_label(target_list)
        Data[name]={
                'Train_Acc': Train_Acc,
                'Train_Loss': Train_Loss,
                'Test_Acc': Test_Acc,
                'Test_Loss':Test_Loss,
             }


X_label='iteration'
test_span=100

figureNo=1
figsize = (8,6)
legend_prop = {'size': 24}
plot_beg=0
plot_stop=600
plt.close('all')

lw=1.2
smooth_size=20
smooth_start_train_loss=3
issmooth=1
ifsave = False

labels=[]
plt.figure(figureNo,figsize=figsize)

ordered_keys = [
    'SGD',
    'Adam beta1:0.0',
    'AMSGrad beta1:0.0',
    'AdaShift op:max n:1 beta1:0.0',
    'AdaShift op:none n:1 beta1:0.0',
]
ordered_keys = sorted(Data.keys())

for key in ordered_keys:
    Train_Loss=Data[key]['Train_Loss']
    y_train_loss=smooth(Train_Loss.reshape(-1)[plot_beg:plot_stop],smooth_size,smooth_start_train_loss) if issmooth else Train_Loss.reshape(-1)[plot_beg:plot_stop]
    X_range_train=np.arange(1,y_train_loss.size+1)*test_span
    if key == 'AdaShift op:none n:1 beta1:0.0':
        log=y_train_loss
    plt.plot(X_range_train,y_train_loss)
    labels.append(key)

plt.xlabel(X_label)
plt.ylabel('training loss')
plt.legend(labels=labels,loc='upper right',prop=legend_prop)
plt.ylim(-50,1000)
plt.tight_layout()

labels=[]
plt.figure(figureNo+1,figsize=figsize)
for key in ordered_keys:
    Test_Acc=Data[key]['Test_Acc']
    smooth_start = 10
    y_test_acc= Test_Acc.reshape(-1)[plot_beg:plot_stop]
    y_test_acc_2=smooth(Test_Acc.reshape(-1)[plot_beg+smooth_start:plot_stop],smooth_size) if 1 else Test_Acc.reshape(-1)[plot_beg+smooth_start:plot_stop]
    y_test_acc_final=np.append(y_test_acc[plot_beg:plot_beg+smooth_start],y_test_acc_2)
    X_range_test=np.arange(1,y_test_acc.size+1)*test_span
    
    plt.plot(X_range_test,y_test_acc_final)
    labels.append(key)
    
plt.xlabel(X_label)
plt.ylabel('test accuracy')
plt.legend(labels=labels,loc='lower right',prop=legend_prop)
plt.ylim(0.4,1)
plt.tight_layout()

plt.show()
    

if ifsave:
    plot_type=input('Input picture name type:')
    plot_dir='/home/victor/projects/Plots/MulitiLayer/%s'%plot_type
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.figure(figureNo)
    plt.savefig('%s/Train_Loss_%s'%(plot_dir,plot_type),format='pdf')

    plt.figure(figureNo+1)
    plt.savefig('%s/Train_Acc_%s'%(plot_dir,plot_type),format='pdf')

    plt.figure(figureNo+2)
    plt.savefig('%s/Test_Loss_%s'%(plot_dir,plot_type),format='pdf')

    plt.figure(figureNo+3)
    plt.savefig('%s/Test_Acc_%s'%(plot_dir,plot_type),format='pdf')
    
    
