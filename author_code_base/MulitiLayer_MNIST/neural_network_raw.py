""" Neural Network.

A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).

Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
import os
from os.path import join, exists
import tensorflow as tf
import numpy as np
import time
from common.utils import allocate_gpu
import argparse
import sys 
sys.path.append("..")
import optimizer_all as optimizer


parser = argparse.ArgumentParser()
parser.add_argument('--GPU', type=int, default=-1, help="which gpu to use")
parser.add_argument('--run_time', type=int, default=-1, help="which time to run this experiment, used in the identifier of experiment. -1 automaticly add one to last time, -2 keep last record")
parser.add_argument('--optimizer_name', type=str, default="adashift", help='sgd | adam | amsgrad | adashift')
parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate')
parser.add_argument("--beta1", type=float, default=0.9, help="beta1 of adam | adashift | amsgrad optimizer")
parser.add_argument("--beta2", type=float, default=0.999, help="beta2 of adam | adashift | amsgrad optimizer")
parser.add_argument("--epsilon", type=float, default=1e-8, help="epsilon of adam | adashift | amsgrad optimizer")
parser.add_argument("--keep_num", type=int, default=10, help="keep_num of adashift optimizer")
parser.add_argument("--pred_g_op", type=str, default="max", help="pred_g_op of adashift optimizer. The operation on the previous gradients. We choose max operation in the paper")

parser.add_argument('--total_epochs', type=int, default=200, help="# of total training epoch")
parser.add_argument('--batch_size', type=int, default=64, help="batch size")
parser.add_argument('--display_step', type=int, default=100, help="frequency to display training statistic")
parser.add_argument('--save_epoch', type=int, default=10, help="frequency to save training statistic")
parser.add_argument('--test_span', type=int, default=100, help="step interval for test")

args = parser.parse_args()

if not exists("./logs"):
    os.makedirs("./logs")

def find_next_time(path_list, default=-1):
    if default > -1:
        return default
    run_times = [int(path.split('_')[0]) for path in path_list]
    # last_time = max(run_times)
    if default == -1:
        next_time = max(run_times) + 1 if run_times else 0
        return next_time
    elif default == -2:
        return max(run_times) if run_times else 0


GPU_ID = allocate_gpu(args.GPU)
print('Using GPU %d' % GPU_ID)
gpuNo = '%d' % GPU_ID

run_time = find_next_time(os.listdir('./logs'), args.run_time)
T = '%d_%s_%s_%d_%.6f_%.2f_%.3f' % (run_time, args.optimizer_name, args.pred_g_op, args.keep_num, args.learning_rate,
                                        args.beta1, args.beta2)
print('Check paras: %s' % T)
if args.run_time == -1:
    time.sleep(5)

log_dir = join("./logs", T)
if not exists(join(log_dir, 'result_data')):
    os.makedirs(join(log_dir, 'result_data'))

total_batch = int(mnist.train.num_examples / args.batch_size)
print('Total batch:%d' % total_batch)

# Network Parameters
n_hidden_1 = 256  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
num_input = 784  # MNIST data input (img shape: 28*28)
num_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes])),
    'h3': tf.Variable(tf.random_normal([num_input, n_hidden_2]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes])),
}


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))

if args.optimizer_name == 'adam':
    optimizer_choise = optimizer.Adam(learning_rate=args.learning_rate, beta1=args.beta1, beta2=args.beta2, epsilon=args.epsilon)
elif args.optimizer_name == 'adashift':
    optimizer_choise = optimizer.AdaShift(learning_rate=args.learning_rate, keep_num=args.keep_num, beta1=args.beta1,
                                                 beta2=args.beta2, epsilon=args.epsilon, pred_g_op=args.pred_g_op)
elif args.optimizer_name == 'amsgrad':
    optimizer_choise = optimizer.AMSGrad(learning_rate=args.learning_rate, beta1=args.beta1, beta2=args.beta2, epsilon=args.epsilon)
elif args.optimizer_name == 'sgd':
    optimizer_choise = optimizer.Grad(learning_rate=args.learning_rate)
else:
    assert 'No optimizer has been chosed, name may be wrong'
train_op = optimizer_choise.minimize(loss_op)
print('Choose Optimizer: %s' % train_op.name)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

Test_Acc = np.zeros((args.total_epochs, total_batch//args.test_span+1))
Test_Loss = np.zeros((args.total_epochs, total_batch//args.test_span+1))
Train_Acc = np.zeros((args.total_epochs, total_batch//args.test_span+1))
Train_Loss = np.zeros((args.total_epochs, total_batch//args.test_span+1))

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    for epoch in range(args.total_epochs):

        for step in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(args.batch_size)
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % args.display_step == 0:
                # Calculate batch loss and accuracy
                train_loss, train_acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                Train_Acc[epoch, step//args.test_span] = train_acc
                Train_Loss[epoch, step//args.test_span] = train_loss

                test_loss, test_acc = sess.run([loss_op, accuracy], feed_dict={X: mnist.test.images, Y: mnist.test.labels})
                Test_Acc[epoch, step//args.test_span] = test_acc
                Test_Loss[epoch, step//args.test_span] = test_loss

                print("[Epoch %d Step %3d/%d]: (%s)(%s_%s)\n  Train Loss:%.4f  Train Acc:%.4f  Test_Loss:%.4f  Test Acc:%.4f"%(
                        epoch, step, total_batch, time.strftime('%H:%M:%S', time.localtime(time.time())), gpuNo, T,
                        train_loss, train_acc, test_loss, test_acc)
                    )

        if epoch % args.save_epoch == 0:
            np.save(join(log_dir, 'result_data', 'Train_Loss.npy'), Train_Loss)
            np.save(join(log_dir, 'result_data', 'Train_Acc.npy'), Train_Acc)
            np.save(join(log_dir, 'result_data', 'Test_Loss.npy'), Test_Loss)
            np.save(join(log_dir, 'result_data', 'Test_Acc.npy'), Test_Acc)

    print("Optimization Finished!")

np.save(join(log_dir, 'result_data', 'Train_Loss.npy'), Train_Loss)
np.save(join(log_dir, 'result_data', 'Train_Acc.npy'), Train_Acc)
np.save(join(log_dir, 'result_data', 'Test_Loss.npy'), Test_Loss)
np.save(join(log_dir, 'result_data', 'Test_Acc.npy'), Test_Acc)

