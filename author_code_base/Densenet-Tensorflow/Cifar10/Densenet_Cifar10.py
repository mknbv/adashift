import os
from os.path import join, exists
gpuNo = os.environ["CUDA_VISIBLE_DEVICES"] = "0"
severNo = 'gpu'
import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
from cifar10 import *
import os
import argparse
import sys 
sys.path.append("../..")
import optimizer_all as optimizer


parser = argparse.ArgumentParser()
parser.add_argument('--run_time', type=int, default=-1, help="which time to run this experiment, used in the identifier of experiment. -1 automaticly add one to last time, -2 keep last record")
parser.add_argument('--growth_k', type=int, default=24, help="growth rate for every layer")
parser.add_argument('--nb_block', type=int, default=2, help="# of dense block + transition layer")
parser.add_argument('--init_learning_rate', type=float, default=0.01, help="initial learning rate")
parser.add_argument('--optimizer_name', type=str, default="adashift", help='sgd | adam | amsgrad | adashift')
parser.add_argument('--beta1', type=float, default=0.9, help="beta1 of optimizer")
parser.add_argument('--beta2', type=float, default=0.999, help="beta2 of optimizer")
parser.add_argument('--epsilon', type=float, default=1e-8, help="epsilon of optimizer")
parser.add_argument('--pred_g_op', type=str, default="max", help="pred_g_op of adashift optimizer")
parser.add_argument('--keep_num', type=int, default=10, help="keep_num of adashift optimizer")
parser.add_argument('--dropout_rate', type=float, default=0.2, help="dropout rate")
parser.add_argument('--batch_size', type=int, default=64, help="batch size")
parser.add_argument('--total_epochs', type=int, default=150, help="# of total training epoch")
parser.add_argument('--random_seed', type=int, default=1, help="random seed")
parser.add_argument('--save_epoch', type=int, default=10, help="frequency to save training statistic 'npy' file")

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

run_time = find_next_time(os.listdir('./logs'), args.run_time)
T = '%d_%s_%s_%d_%.6f_%.2f_%.3f' % (run_time, args.optimizer_name, args.pred_g_op, args.keep_num, args.init_learning_rate,
                                        args.beta1, args.beta2)
print('Check paras: %s' % T)
if args.run_time == -1:
    time.sleep(5)

log_dir = join("./logs", T)
if not exists(join(log_dir, 'result_data')):
    os.makedirs(join(log_dir, 'result_data'))

# log_dir = './logs/%s_%s_%d_%.3f_%.2f_%.3f' % (args.T, args.pred_g_op, args.keep_num, args.init_learning_rate,
#                                               args.beta1, args.beta2)
# checkpoint_dir = './model/model-%s_%s_%d_%.3f_%.2f_%.3f' % (args.T, args.pred_g_op, args.keep_num,
#                                                             args.init_learning_rate, args.beta1, args.beta2)
checkpoint_dir = join("./logs", T)
if not os.path.exists(log_dir+'/result_data'):
    os.makedirs(log_dir+'/result_data')

np.random.seed(args.random_seed)
tf.set_random_seed(args.random_seed)

class_num = 10
test_iteration = 10


def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
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

def Evaluate(sess):
    test_acc = 0.0
    test_loss = 0.0
    test_pre_index = 0
    add = 1000

    for it in range(test_iteration):
        test_batch_x = test_x[test_pre_index: test_pre_index + add]
        test_batch_y = test_y[test_pre_index: test_pre_index + add]
        test_pre_index = test_pre_index + add

        test_feed_dict = {
            x: test_batch_x,
            label: test_batch_y,
            learning_rate: epoch_learning_rate,
            training_flag: False
        }

        loss_, acc_ = sess.run([cost, accuracy], feed_dict=test_feed_dict)

        test_loss += loss_ / 10.0
        test_acc += acc_ / 10.0

    summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])

    return test_acc, test_loss, summary

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
        # x = Max_Pooling(x, pool_size=[3,3], stride=2)
        """
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

        x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_final')

        # 100 Layer
        x = Batch_Normalization(x, training=self.training, scope='linear_batch')
        x = Relu(x)
        x = Global_Average_Pooling(x)
        x = flatten(x)
        x = Linear(x)

        # x = tf.reshape(x, [-1, 10])
        return x


train_x, train_y, test_x, test_y = prepare_data()
train_x, test_x = color_preprocessing(train_x, test_x)

iteration = train_x.shape[0]//args.batch_size

# image_size = 32, img_channels = 3, class_num = 10 in cifar10
x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, img_channels])
label = tf.placeholder(tf.float32, shape=[None, class_num])
training_flag = tf.placeholder(tf.bool)
learning_rate = tf.placeholder(tf.float32, name='learning_rate')

logits = DenseNet(x=x, nb_blocks=args.nb_block, filters=args.growth_k, training=training_flag).model
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))


if args.optimizer_name == 'adam':
    optimizer_choise = optimizer.Adam(learning_rate=learning_rate,beta1=args.beta1,beta2=args.beta2,epsilon=args.epsilon)
elif args.optimizer_name == 'adashift':
    optimizer_choise = optimizer.AdaShift(learning_rate=learning_rate, keep_num=args.keep_num, beta1=args.beta1,
                                             beta2=args.beta2, epsilon=args.epsilon, pred_g_op=args.pred_g_op)
elif args.optimizer_name == 'amsgrad':
    optimizer_choise = optimizer.AMSGrad(learning_rate=learning_rate, beta1=args.beta1, beta2=args.beta2, epsilon=args.epsilon)
elif args.optimizer_name == 'sgd':
    optimizer_choise = optimizer.Grad(learning_rate=learning_rate)
else:
    assert 'No optimizer has been chosed, name may be wrong'

train = optimizer_choise.minimize(cost)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(tf.global_variables())
config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

    Test_Acc = np.zeros((args.total_epochs+1,1)) if not os.path.exists(log_dir+'/result_data/Test_Acc.npy') else np.load(log_dir+'/result_data/Test_Acc.npy')
    Test_Loss= np.zeros((args.total_epochs+1,1)) if not os.path.exists(log_dir+'/result_data/Test_Loss.npy') else np.load(log_dir+'/result_data/Test_Loss.npy')
    Train_Acc= np.zeros((args.total_epochs+1,1)) if not os.path.exists(log_dir+'/result_data/Train_Acc.npy') else np.load(log_dir+'/result_data/Train_Acc.npy')
    Train_Loss=np.zeros((args.total_epochs+1,1)) if not os.path.exists(log_dir+'/result_data/Train_Loss.npy') else np.load(log_dir+'/result_data/Train_Loss.npy')
    epoch_learning_rate = args.init_learning_rate
    for epoch in range(1, args.total_epochs + 1):
        if epoch == (args.total_epochs * 0.5) or epoch == (args.total_epochs * 0.75):
            epoch_learning_rate = epoch_learning_rate / 10

        pre_index = 0
        train_acc = 0.0
        train_loss = 0.0

        for step in range(1, iteration + 1):
            if pre_index+args.batch_size < 50000 :
                batch_x = train_x[pre_index : pre_index+args.batch_size]
                batch_y = train_y[pre_index : pre_index+args.batch_size]
            else :
                batch_x = train_x[pre_index : ]
                batch_y = train_y[pre_index : ]

            batch_x = data_augmentation(batch_x)

            train_feed_dict = {
                x: batch_x,
                label: batch_y,
                learning_rate: epoch_learning_rate,
                training_flag : True
            }

            _, batch_loss = sess.run([train, cost], feed_dict=train_feed_dict)
            batch_acc = accuracy.eval(feed_dict=train_feed_dict)

            train_loss += batch_loss
            train_acc += batch_acc
            pre_index += args.batch_size

            if step == iteration :
                train_loss /= iteration # average loss
                train_acc /= iteration # average accuracy

                train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                                                  tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])

                test_acc, test_loss, test_summary = Evaluate(sess)

                summary_writer.add_summary(summary=train_summary, global_step=epoch)
                summary_writer.add_summary(summary=test_summary, global_step=epoch)
                summary_writer.flush()

                line = "Epoch: %d/%d %s(%s) \n  train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f " % (
                    epoch, args.total_epochs, T+' '+severNo+'_'+gpuNo,time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), train_loss, train_acc, test_loss, test_acc)
                print(line)

                with open('logs.txt', 'a') as f :
                    f.write(line)

                Test_Acc[epoch]=test_acc
                Test_Loss[epoch]=test_loss
                Train_Acc[epoch]=train_acc
                Train_Loss[epoch]=train_loss

        if epoch % args.save_epoch == 0:
            np.save(log_dir+'/result_data/Train_Loss.npy',Train_Loss)
            np.save(log_dir+'/result_data/Train_Acc.npy',Train_Acc)
            np.save(log_dir+'/result_data/Test_Loss.npy',Test_Loss)
            np.save(log_dir+'/result_data/Test_Acc.npy',Test_Acc)
        saver.save(sess=sess, save_path=checkpoint_dir+'/dense.ckpt')

    np.save(log_dir+'/result_data/Train_Loss',Train_Loss)
    np.save(log_dir+'/result_data/Train_Acc',Train_Acc)
    np.save(log_dir+'/result_data/Test_Loss',Test_Loss)
    np.save(log_dir+'/result_data/Test_Acc',Test_Acc)