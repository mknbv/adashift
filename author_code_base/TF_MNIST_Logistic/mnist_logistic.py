#logistic_regression by ffzhang
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]='2'
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
from matplotlib import pyplot as plt
import matplotlib
from scipy.signal import savgol_filter
import sys 
sys.path.append("..")
from optimizer_all import AdaShift,Adam,AMSGrad,Grad

font = {'family' : 'monospace',
        'size'   : 24}

matplotlib.rc('font', **font)

def map_label(key):
    key_list=key.split('_')
    if key.startswith('AdaShift'):
        return 'AdaShift op:%s n:%s beta1:%s.%s'%(key_list[1],key_list[2],key_list[3],key_list[4])
    elif key.startswith('Adam'):
        return 'Adam beta1:%s.%s'%(key_list[1],key_list[2])
    elif key.startswith('AMSGrad'):
        return 'AMSGrad beta1:%s.%s'%(key_list[1],key_list[2])
    elif key.startswith('SGD'):
        return 'SGD'
    else:
        print('No Mapping')
        return key


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



        

mnist=input_data.read_data_sets('data/mnist',one_hot=True)

n_epochs = 20
batch_size=128
learning_rate=0.001
learning_rate_max=0.01
beta_1=0.9
beta_2=0.999
epsilon=1e-10
pred_g_op='max'
keep_num=10

log_dir = './data/log_2'
test_gap=20

np.random.seed(10)
tf.set_random_seed(10)
X = tf.placeholder(tf.float32,[None,784],name='X_placeholder')
Y = tf.placeholder(tf.int32, [None,10],name='Y_placehoder')
test_dict={X:mnist.test.images, Y:mnist.test.labels}

w = tf.Variable(tf.random_normal(shape=[784,10],stddev=0.01),name='weights')
b = tf.Variable(tf.zeros([1,10]),name='bias')

# W*x+b
logits=tf.matmul(X,w)+b#+0.01*(tf.reduce_mean(tf.abs(w))+tf.abs(b))
# logits=tf.matmul(X,w)+b+0.01*(tf.reduce_mean(tf.abs(w))+tf.abs(b))

entropy=tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y,name='entropy')
loss=tf.reduce_mean(entropy,name='loss')
tf.summary.scalar('loss', loss)

mystep=tf.Variable(0,trainable=False)
#optimizer = AdamOptimizer(learning_rate,epsilon=epsilon).minimize(loss,global_step=tf.train.create_global_step())
optimizer1 = Grad(learning_rate=0.1,name='SGD').minimize(loss,global_step=mystep)
optimizer2 = Adam(learning_rate,beta1=0.9,beta2=0.999,epsilon=epsilon,name='Adam_0_9').minimize(loss,global_step=mystep)
optimizer3 = Adam(learning_rate,beta1=0.0,beta2=0.999,epsilon=epsilon,name='Adam_0_0').minimize(loss,global_step=mystep)
optimizer4 = AMSGrad(learning_rate, beta1= 0.9,beta2=0.999, epsilon= epsilon,name='AMSGrad_0_9').minimize(loss,global_step=mystep)
optimizer5 = AMSGrad(learning_rate, beta1= 0.0,beta2=0.999, epsilon= epsilon,name='AMSGrad_0_0').minimize(loss,global_step=mystep)
optimizer6 = AdaShift(learning_rate=learning_rate_max,keep_num=1, beta1=0.0,beta2=0.999,pred_g_op='max',epsilon=epsilon,name='AdaShift_max_1_0_0').minimize(loss,global_step=mystep)
optimizer7 = AdaShift(learning_rate=learning_rate_max,keep_num=10,beta1=0.9,beta2=0.999,pred_g_op='max',epsilon=epsilon,name='AdaShift_max_10_0_9').minimize(loss,global_step=mystep)
optimizer8 = AdaShift(learning_rate=learning_rate_max,keep_num=10,beta1=1.0,beta2=0.999,pred_g_op='max',epsilon=epsilon,name='AdaShift_max_10_1_0').minimize(loss,global_step=mystep)
optimizer9 = AdaShift(learning_rate=learning_rate, keep_num=1, beta1=0.0,beta2=0.999,pred_g_op='none',epsilon=epsilon,name='AdaShift_none_1_0_0').minimize(loss,global_step=mystep)
optimizer10 =AdaShift(learning_rate=learning_rate, keep_num=10,beta1=0.9,beta2=0.999,pred_g_op='none',epsilon=epsilon,name='AdaShift_none_10_0_9').minimize(loss,global_step=mystep)
optimizer11 = Adam(learning_rate,beta1=0.999,beta2=0.999,epsilon=epsilon,name='Adam_0_999').minimize(loss,global_step=mystep)
optimizer12 = Adam(learning_rate,beta1=0.9999,beta2=0.999,epsilon=epsilon,name='Adam_0_9999').minimize(loss,global_step=mystep)

Optimizers=[
   optimizer1,
#    optimizer2,
   optimizer3,
#    optimizer4,
   optimizer5,
   optimizer6,
   optimizer7,
#    optimizer8,
   optimizer9,
#    optimizer10,

###big beta1
##    optimizer11,
##    optimizer12,
 ]

Test,Loss={},{}

for opNo,optimizer in enumerate(Optimizers):
    Y_ = tf.nn.softmax(logits)
    correct_preds=tf.equal(tf.argmax(Y_,1),tf.argmax(Y,1),name='test_correct_tensor')
    accuracy=tf.reduce_mean(tf.cast(correct_preds,tf.float32),name='test_accuracy')
    tf.summary.scalar('accuracy', accuracy)


    init=tf.global_variables_initializer()

    with tf.Session() as sess:
        start_time=time.time()
        sess.run(init)
        n_batches=int(mnist.train.num_examples/batch_size)

        Test_Acc=np.zeros((n_epochs,n_batches//test_gap+1))
        loss_np=np.zeros((n_epochs,n_batches))
        for i in range(n_epochs):

            total_loss=0
            for j in range(n_batches):
                X_batch, Y_batch =mnist.train.next_batch(batch_size)
                optzer,loss_batch =sess.run([optimizer,loss],feed_dict={X:X_batch,Y:Y_batch})
                loss_np[i,j]=loss_batch
                total_loss +=loss_batch
                if j%test_gap==0:
                    test_acc=sess.run([accuracy],feed_dict=test_dict)
                    Test_Acc[i,j//test_gap]=test_acc[0]
            print ('Average training loss epoch {0}:{1}'.format(i,total_loss/n_batches))
        print ('Total time: {0} seconds'.format(time.time()-start_time))
        print ('optimizatin Finished')
        name=map_label(optimizer.name)
        Test[name],Loss[name]=Test_Acc,loss_np


color_map={
        'SGD':                              '#BBBB00',
        'Adam beta1:0.0':                   '#0066FF',     
        'Adam beta1:0.9':                   '#003C9D',  
        'AMSGrad beta1:0.0':                '#00DD00',
        'AMSGrad beta1:0.9':                '#008844',
        'AdaShift op:max n:1 beta1:0.0':    '#FF3333',
        'max-AdaShift op:max n:1 beta1:0.0':    '#FF3333',
        'AdaShift op:max n:10 beta1:0.9':   '#CC0000',
        'AdaShift op:max n:10 beta1:1.0':   '#AA0000',
        'AdaShift op:none n:1 beta1:0.0':   '#EE7700',
        'non-AdaShift op:none n:1 beta1:0.0':   '#EE7700',
        'AdaShift op:none n:10 beta1:0.9':  '#BB5500',
        'AdaShift op:none n:10 beta1:1.0':  '#A42D00',
    }

ordered_keys = [
    'SGD',
    'Adam beta1:0.0',
    'AMSGrad beta1:0.0',
    'AdaShift op:max n:1 beta1:0.0',
    'AdaShift op:none n:1 beta1:0.0',
]

# for optimizer with momentum
#ordered_keys = [
#    'Adam beta1:0.9',
#    'AMSGrad beta1:0.9',
#    'AdaShift op:max n:10 beta1:0.9',
#    'AdaShift op:none n:10 beta1:0.9',
#]



lw=1.2
legend_prop = {'size': 24}
smoothbox=50
issmooth=1
labels=[]
plt.close('all')
plt.figure(3,figsize=(8,6))

for key in ordered_keys:
    print('Plot test %s'%key)
    y_test=smooth(Test[key].reshape(-1),smoothbox//10) if 0 else Test[key].reshape(-1)
    X_range_train=np.arange(1,Test[key].size+1)*test_gap
    plt.plot(X_range_train,y_test,lw=lw,color=color_map[key])
    label = key.split(' ')[0] if key.split(' ')[0] != 'AdaShift' else key.split(' ')[0]+' '+key.split(' ')[1]

    labels.append(key)

plt.ylim(0.4,0.95)
plt.ylabel('test accuracy')
plt.xlabel('iterations')
plt.legend(labels,loc='lower right',prop=legend_prop)
plt.show()
plt.tight_layout()

labels=[]
plt.figure(4,figsize=(8,6))
for key in ordered_keys:
    print('Plot train %s'%key)
    X_range_test=np.arange(1,Loss[key].size+1)*test_gap
    y_train=smooth(Loss[key].reshape(-1),smoothbox) if issmooth else Loss[key].reshape(-1)
    plt.plot(X_range_test,y_train,lw=lw,color=color_map[key])
    label = key.split(' ')[0] if key.split(' ')[0] != 'AdaShift' else key.split(' ')[0]+' '+key.split(' ')[1]
    labels.append(key)
    # labels.append(key.split(' ')[0])

plt.ylim(0,2)
plt.ylabel('train loss')
plt.xlabel('iterations')
plt.legend(labels,loc='upper right',prop=legend_prop)
plt.show()
plt.tight_layout()