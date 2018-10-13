from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
import os
import tensorflow as tf
import numpy as np
import time
import optimizer_all
from os.path import join, exists
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--update', type=int, default=0, help='set to 0, not update gradient within the last several epochs;'
                                                          'set to 1, update gradient within the last several epochs')
parser.add_argument('--start_epoch', type=int, default=20, help='gradient from which epoch involved in calculation')
parser.add_argument('--end_epoch', type=int, default=30, help='gradient until which epoch involved in calculation')
parser.add_argument('--exp_name', type=str, default="", help='name(identifier) of experiment')
parser.add_argument('--spatial_pair_num', type=int, default=256, help='# of variable pairs used when calculate spatial correlation coefficient')
parser.add_argument('--random', type=str, default="", help='normal | uniform | "";'
                                                           'if not empty, generate gradients sampling from specified random distribution')
parser.add_argument('--optimizer_name', type=str, default="adaShift", help='sgd | adam | amsgrad | adaShift')
parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate')
parser.add_argument("--beta1", type=float, default=0.9, help="beta1 of adam | adashift | amsgrad optimizer")
parser.add_argument("--beta2", type=float, default=0.999, help="beta2 of adam | adashift | amsgrad optimizer")
parser.add_argument("--epsilon", type=float, default=1e-8, help="epsilon of adam | adashift | amsgrad optimizer")
parser.add_argument("--keep_num", type=int, default=40, help="keep_num of adashift optimizer")
parser.add_argument("--pred_g_op", type=str, default="max", help="pred_g_op of adashift optimizer")
parser.add_argument("--use_mov", type=int, default=1, help="set to 0, not use move in adashift optimizer; "
                                                           "set to 1, use move in adashift optimizer")
parser.add_argument("--mov_num", type=int, default=30, help="mov_num of adashift optimizer")
parser.add_argument('--batch_size', type=int, default=64, help="batch size")
parser.add_argument('--display_step', type=int, default=100, help="frequency to display training statistic")

parser.add_argument("--beta2_calc_adam", type=float, default=0.9, help="beta2 for calculating vt in adam")
parser.add_argument("--beta2_calc_adashift", type=float, default=0.999, help="beta2 for calculating vt in adashift")

args = parser.parse_args()


def correlation(seq1, seq2):
    x = seq1.reshape(-1)
    y = seq2.reshape(-1)
    assert len(x) == len(y), "correlation: length not the same"

    x_mean = x.mean()
    y_mean = y.mean()
    cor = ((x-x_mean)*(y-y_mean)).sum() / np.sqrt(((x-x_mean)**2).sum() * ((y-y_mean)**2).sum())

    return cor


def correlation_arr(arr1, arr2):
    assert arr1.shape[0]==arr2.shape[0] and arr1.shape[1]==arr2.shape[1], "correlation_arr: shape not the same"

    x = arr1
    y = arr2
    x_mean = np.tile(x.mean(axis=0).reshape(1, -1), (arr1.shape[0], 1))
    y_mean = np.tile(y.mean(axis=0).reshape(1, -1), (arr2.shape[0], 1))
    cor = ((x-x_mean)*(y-y_mean)).sum(axis=0) / np.sqrt(((x-x_mean)**2).sum(axis=0) * ((y-y_mean)**2).sum(axis=0))

    return cor


total_batch = int(mnist.train.num_examples / args.batch_size)
print('Total batch:%d' % total_batch)

log_dir = join('./logs', args.exp_name)
if not exists(log_dir):
    os.makedirs(log_dir)

# Network Parameters
n_hidden_1 = 256  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
num_input = 784  # MNIST data input (img shape: 28*28)
num_classes = 10  # MNIST total classes (0-9 digits)

gt_list = np.zeros((total_batch*args.end_epoch, n_hidden_1*n_hidden_2))

if args.random != "":
    if args.random == "normal":
        gt_list = np.random.normal(loc=0, scale=1, size=gt_list.shape)
    elif args.random == "uniform":
        gt_list = np.random.uniform(-1, 1, size=gt_list.shape)
    else:
        raise Exception("undefined distribution")
elif exists(join(log_dir, "gt.npy")):
    print("load from disk")
    gt_list = np.load(join(log_dir, "gt.npy"))
else:
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

    g_t = tf.gradients(loss_op, weights['h2'])

    if args.optimizer_name == 'adam':
        optimizer_choise = optimizer_all.Adam(learning_rate=args.learning_rate, beta1=args.beta1, beta2=args.beta2, epsilon=args.epsilon)
    elif args.optimizer_name == 'adaShift':
        optimizer_choise = optimizer_all.AdaShift(learning_rate=args.learning_rate, keep_num=args.keep_num, beta1=args.beta1,
                                                  beta2=args.beta2,
                                                  use_mov=(args.use_mov == 1), mov_num=args.mov_num, epsilon=args.epsilon,
                                                  pred_g_op=args.pred_g_op)
    elif args.optimizer_name == 'amsgrad':
        optimizer_choise = optimizer_all.AMSGrad(learning_rate=args.learning_rate, beta1=args.beta1, beta2=args.beta2, epsilon=args.epsilon)
    elif args.optimizer_name == 'sgd':
        optimizer_choise = optimizer_all.Grad(learning_rate=args.learning_rate)
    else:
        assert 'No optimizer has been chosed, name may be wrong'
    train_op = optimizer_choise.minimize(loss_op)
    print('Choose Optimizer: %s' % train_op.name)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)
        for epoch in range(args.end_epoch):
            for step in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(args.batch_size)

                _gt = sess.run(g_t, feed_dict={X: batch_x, Y: batch_y})
                _gt = np.reshape(_gt, [-1])

                gt_list[epoch*total_batch+step] = _gt

                if not (epoch >= args.start_epoch and args.update == 0):
                    sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

                if step % args.display_step == 0:
                    # Calculate batch loss and accuracy
                    train_loss, train_acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                         Y: batch_y})

                    test_loss, test_acc = sess.run([loss_op, accuracy], feed_dict={X: mnist.test.images,Y: mnist.test.labels})

                    print("[Epoch %d Step %3d/%d]: (%s)\n  Train Loss:%.4f  Train Acc:%.4f  Test_Loss:%.4f  Test Acc:%.4f" % (
                            epoch, step, total_batch, time.strftime('%H:%M:%S', time.localtime(time.time())),
                            train_loss, train_acc, test_loss, test_acc)
                        )

        print("Optimization Finished!")
    np.save(join(log_dir, "gt.npy"), gt_list)

print("start to calc")

## temporal gt
cor_offset_mean = []
cor_offset_std = []
for offset in range(1, 11):
    cors = correlation_arr(gt_list[args.start_epoch*total_batch:gt_list.shape[0]-offset, :],
                                      gt_list[args.start_epoch*total_batch+offset:, :])
    cor_offset_mean.append(cors.mean())
    cor_offset_std.append(cors.std())
print("temporal gt cors:")
print(list(range(1, 11)))
print(cor_offset_mean)
print(cor_offset_std)


## spatial gt
k_kks = np.random.randint(0, gt_list.shape[1], size=(args.spatial_pair_num*2, 2))
k_kks_tmp = []
for i in range(k_kks.shape[0]):
    if len(k_kks_tmp) >= args.spatial_pair_num:
        break
    if k_kks[i][0] != k_kks[i][1]:
        k_kks_tmp.append(k_kks[i:i+1, :])
k_kks = np.concatenate(k_kks_tmp, axis=0)
print(args.spatial_pair_num, k_kks.shape[0])
cor_offset_mean = []
cor_offset_std = []
gt_list_select_k = gt_list[args.start_epoch*total_batch:args.end_epoch*total_batch, k_kks[:, 0]]
gt_list_select_kk = gt_list[args.start_epoch*total_batch:args.end_epoch*total_batch, k_kks[:, 1]]
print("finished prepare k_kks array")
for offset in range(1, 11):
    cors = correlation_arr(gt_list_select_k[:gt_list_select_k.shape[0]-offset], gt_list_select_kk[offset:])
    cor_offset_mean.append(cors.mean())
    cor_offset_std.append(cors.std())
print("spatial gt cors:")
print(list(range(1, 11)))
print(cor_offset_mean)
print(cor_offset_std)


## gt && vt adam
beta2_adam = args.beta2_calc_adam
_vt_none = np.zeros_like(gt_list[0])
vt_none_list = np.zeros((total_batch * args.end_epoch, n_hidden_1 * n_hidden_2))
for j in range(gt_list.shape[0]):
    _vt_none = _vt_none * beta2_adam + (1 - beta2_adam) * (gt_list[j] ** 2)
    _vt_debias_none = _vt_none
    vt_none_list[j] = _vt_debias_none

cors_none = correlation_arr(gt_list[args.start_epoch * total_batch:args.end_epoch * total_batch, :] ** 2,
                            vt_none_list[args.start_epoch * total_batch:args.end_epoch * total_batch, :])

print("gt && vt adam cors: ", cors_none.mean(), cors_none.std())


## gt && vt adamshift(none & max)
cors_none_mean = []
cors_none_std = []
cors_max_mean = []
cors_max_std = []
for i in range(1, 11):
    _vt_none = np.zeros_like(gt_list[0])
    _vt_max = np.zeros_like(gt_list[0])
    vt_none_list = np.zeros((total_batch * args.end_epoch, n_hidden_1 * n_hidden_2))
    vt_max_list = np.zeros((total_batch * args.end_epoch, n_hidden_1 * n_hidden_2))

    for j in range(i, gt_list.shape[0]):
        _vt_none = _vt_none * args.beta2_calc_adashift + (1 - args.beta2_calc_adashift) * (gt_list[j-i] ** 2)
        _vt_debias_none = _vt_none

        _vt_max = _vt_max * args.beta2_calc_adashift + (1 - args.beta2_calc_adashift) * np.max(gt_list[j-i] ** 2)
        _vt_debias_max = _vt_max

        vt_none_list[j] = _vt_debias_none
        vt_max_list[j] = _vt_debias_max

    cors_none = correlation_arr(gt_list[args.start_epoch*total_batch:args.end_epoch*total_batch, :] ** 2,
                                vt_none_list[args.start_epoch*total_batch:args.end_epoch*total_batch, :])
    cors_max = correlation_arr(gt_list[args.start_epoch*total_batch:args.end_epoch*total_batch, :] ** 2,
                                vt_max_list[args.start_epoch*total_batch:args.end_epoch*total_batch, :])
    cors_none_mean.append(cors_none.mean())
    cors_none_std.append(cors_none.std())
    cors_max_mean.append(cors_max.mean())
    cors_max_std.append(cors_max.std())

print("gt && vt adashift none cors:")
print(cors_none_mean)
print(cors_none_std)
print("gt && vt adashift max cors:")
print(cors_max_mean)
print(cors_max_std)


