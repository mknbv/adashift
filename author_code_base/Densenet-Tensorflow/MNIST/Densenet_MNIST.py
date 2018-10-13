import os
gpuNo=os.environ["CUDA_VISIBLE_DEVICES"] = "0"
severNo='gpu'
import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import optimizer
import numpy as np
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--T', type=str, default="18_adam", help="identifier of experiment")
parser.add_argument('--growth_k', type=int, default=12, help="growth rate for every layer")
parser.add_argument('--nb_block', type=int, default=2, help="# of dense block + transition layer")
parser.add_argument('--init_learning_rate', type=float, default=0.01, help="initial learning rate")
parser.add_argument('--optimizer_name', type=str, default="adamshiftmoving", help='sgd | adam | amsgrad | adashift')
parser.add_argument('--beta1', type=float, default=0.9, help="beta1 of optimizer")
parser.add_argument('--beta2', type=float, default=0.999, help="beta2 of optimizer")
parser.add_argument('--epsilon', type=float, default=1e-5, help="epsilon of optimizer")
parser.add_argument('--pred_g_op', type=str, default="none", help="pred_g_op of adashift optimizer")
parser.add_argument('--keep_num', type=int, default=20, help="keep_num of adashift optimizer")
parser.add_argument('--dropout_rate', type=float, default=0.2, help="dropout rate")
parser.add_argument('--batch_size', type=int, default=100, help="batch size")
parser.add_argument('--total_epochs', type=int, default=20, help="# of total training epoch")
parser.add_argument('--random_seed', type=int, default=1, help="random seed")
parser.add_argument('--save_epoch', type=int, default=10, help="frequency to save training statistic")
parser.add_argument('--test_span', type=int, default=50, help="step interval for test")

args = parser.parse_args()

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
class_num = 10
total_batch = int(mnist.train.num_examples / args.batch_size)

log_dir='./logs/%s_%d_%s_%3f_%.3f_%.3f' % (args.T, args.keep_num, args.pred_g_op, args.init_learning_rate,
                                           args.beta1, args.beta2)
checkpoint_dir='./model/model_%s' % args.T
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(log_dir+'/result_data'):
    os.makedirs(log_dir+'/result_data')

np.random.seed(args.random_seed)
tf.set_random_seed(args.random_seed)


def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
        return network

def Global_Average_Pooling(x, stride=1):
    """
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter
    It is global average pooling without tflearn
    """

    return global_avg_pool(x, name='Global_avg_pooling')
    # But maybe you need to install h5py and curses or not


def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Drop_out(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Relu(x):
    return tf.nn.relu(x)

def Average_pooling(x, pool_size=[2,2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_Pooling(x, pool_size=[3,3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Linear(x) :
    return tf.layers.dense(inputs=x, units=class_num, name='linear')



class DenseNet():
    def __init__(self, x, nb_blocks, filters, training):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.training = training
        self.model = self.Dense_net(x)


    def bottleneck_layer(self, x, scope):
        # print(x)
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=args.dropout_rate, training=self.training)

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3,3], layer_name=scope+'_conv2')
            x = Drop_out(x, rate=args.dropout_rate, training=self.training)

            # print(x)

            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=args.dropout_rate, training=self.training)
            x = Average_pooling(x, pool_size=[2,2], stride=2)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = Concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)

            x = Concatenation(layers_concat)

            return x

    def Dense_net(self, input_x):
        x = conv_layer(input_x, filter=2 * self.filters, kernel=[7,7], stride=2, layer_name='conv0')
        x = Max_Pooling(x, pool_size=[3,3], stride=2)

        for i in range(self.nb_blocks) :
            # 6 -> 12 -> 48
            x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_'+str(i))
            x = self.transition_layer(x, scope='trans_'+str(i))

        """
        x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_1')
        x = self.transition_layer(x, scope='trans_1')

        x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
        x = self.transition_layer(x, scope='trans_2')

        x = self.dense_block(input_x=x, nb_layers=48, layer_name='dense_3')
        x = self.transition_layer(x, scope='trans_3')
        """

        x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_final')

        # 100 Layer
        x = Batch_Normalization(x, training=self.training, scope='linear_batch')
        x = Relu(x)
        x = Global_Average_Pooling(x)
        x = flatten(x)
        x = Linear(x)

        # x = tf.reshape(x, [-1, 10])
        return x


x = tf.placeholder(tf.float32, shape=[None, 784])
batch_images = tf.reshape(x, [-1, 28, 28, 1])
label = tf.placeholder(tf.float32, shape=[None, 10])
training_flag = tf.placeholder(tf.bool)
learning_rate = tf.placeholder(tf.float32, name='learning_rate')

logits = DenseNet(x=batch_images, nb_blocks=args.nb_block, filters=args.growth_k, training=training_flag).model
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

optimizer_choise = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=args.beta1,beta2=args.beta2, epsilon=args.epsilon)
# optimizer_choise = optimizer.AdamShiftN(learning_rate=learning_rate,keep_num=keep_num,beta2=beta2, epsilon=epsilon,pred_g_op=pred_g_op)
train = optimizer_choise.minimize(cost)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', accuracy)

saver = tf.train.Saver(tf.global_variables())

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, intra_op_parallelism_threads=0, inter_op_parallelism_threads=0)
config.gpu_options.allow_growth = True
sess =  tf.Session(config=config)

ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(log_dir, sess.graph)

global_step = 0
epoch_learning_rate = args.init_learning_rate
test_feed_dict = {
    x: mnist.test.images,
    label: mnist.test.labels,
    learning_rate: epoch_learning_rate,
    training_flag : False
}

Test_Acc=np.zeros((args.total_epochs,total_batch//args.test_span+1)) if not os.path.exists(log_dir+'/result_data/Test_Acc.npy')   else np.load(log_dir+'/result_data/Test_Acc.npy')
Test_Loss=np.zeros((args.total_epochs,total_batch//args.test_span+1)) if not os.path.exists(log_dir+'/result_data/Test_Loss.npy') else np.load(log_dir+'/result_data/Test_Loss.npy')
Train_Acc=np.zeros((args.total_epochs,total_batch//args.test_span+1)) if not os.path.exists(log_dir+'/result_data/Train_Acc.npy') else np.load(log_dir+'/result_data/Train_Acc.npy')
Train_Loss=np.zeros((args.total_epochs,total_batch//args.test_span+1)) if not os.path.exists(log_dir+'/result_data/Train_Loss.npy') else np.load(log_dir+'/result_data/Train_Loss.npy')

for epoch in range(args.total_epochs):
    if epoch == (args.total_epochs * 0.5) or epoch == (args.total_epochs * 0.75):
        epoch_learning_rate = epoch_learning_rate / 10

    for step in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(args.batch_size)

        train_feed_dict = {
            x: batch_x,
            label: batch_y,
            learning_rate: epoch_learning_rate,
            training_flag : True
        }

        _, train_loss = sess.run([train, cost], feed_dict=train_feed_dict)

        if step % args.test_span == 0:
            global_step += 100
            train_summary, train_accuracy = sess.run([merged, accuracy], feed_dict=train_feed_dict)\

            test_accuracy,test_loss = sess.run([accuracy,cost], feed_dict=test_feed_dict)
            test_summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                              tf.Summary.Value(tag='test_accuracy', simple_value=test_accuracy)])
            writer.add_summary(train_summary, global_step=epoch*total_batch+step)
            writer.add_summary(test_summary, global_step=epoch*total_batch+step)
            writer.flush()

            print("[Epoch %d Step %3d/%d]: (%s)(%s_%s_%s)\n  Train Loss:%.4f  Train Acc:%.4f  Test_Loss:%.4f  Test Acc:%.4f"%(
                    epoch,step,total_batch,time.strftime('%H:%M:%S',time.localtime(time.time())),gpuNo,severNo,args.T,
                    train_loss,train_accuracy,test_loss,test_accuracy)
                )

            Test_Acc[epoch,step//args.test_span]=test_accuracy
            Test_Loss[epoch,step//args.test_span]=test_loss
            Train_Acc[epoch,step//args.test_span]=train_accuracy
            Train_Loss[epoch,step//args.test_span]=train_loss

    if epoch % args.save_epoch == 0:
        np.save(log_dir+'/result_data/Train_Loss.npy',Train_Loss)
        np.save(log_dir+'/result_data/Train_Acc.npy',Train_Acc)
        np.save(log_dir+'/result_data/Test_Loss.npy',Test_Loss)
        np.save(log_dir+'/result_data/Test_Acc.npy',Test_Acc)

np.save(log_dir+'/result_data/Train_Loss',Train_Loss)
np.save(log_dir+'/result_data/Train_Acc',Train_Acc)
np.save(log_dir+'/result_data/Test_Loss',Test_Loss)
np.save(log_dir+'/result_data/Test_Acc',Test_Acc)
saver.save(sess=sess, save_path=checkpoint_dir+'/dense.ckpt')

sess.close()