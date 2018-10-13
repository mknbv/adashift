import sys, locale, time, os
from os import path

locale.setlocale(locale.LC_ALL, '')
sys.path.append(path.dirname(path.abspath(__file__)))
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
SOURCE_DIR = path.dirname(path.dirname(path.abspath(__file__))) + '/'

import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common.ops import *
from common.score import *
from common.data_loader import *
from common.logger import Logger
# from common.optimizer import *
# import optimizer_shift
import argparse
import sys
sys.path.append("../..")
import optimizer_all as optimizer_shift

cfg = tf.app.flags.FLAGS

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

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=0.00001, help="initial learning rate")
parser.add_argument('--optimizer_name', type=str, default="adashift", help='sgd | adam | amsgrad | adashift')
args = parser.parse_args()

GPU = -1
default_run_time = -1
if not os.path.exists('../result'):
    os.makedirs('../result')
run_time=find_next_time(os.listdir('../result/'),default_run_time)
optimizer_name = args.optimizer_name #adam, sgd, amsgrad, adashift"
learning_rate=args.learning_rate
print(type(optimizer_name),optimizer_name)
beta_1=0.
beta_2=0.999
pred_g_op='max'
epsilon=1e-8
keep_num=1
iMaxIter=150000



T='%d_%s_%s_%d_%.6f_%.2f_%.3f'%(run_time,optimizer_name, pred_g_op,keep_num,learning_rate,beta_1,beta_2)
print('Check paras: %s'%T)
if default_run_time == -1:
    time.sleep(6)

tf.app.flags.DEFINE_string("sResultTag", "ini0.01_wgans_maxgp0.1_relu_max0.9_0.99_bs64_lr0.0001_0.0001_128fres2k3act0upconv_128fres2k3act0downconv_sphz100", "your tag for each test case")

tf.app.flags.DEFINE_integer("iTrainG", 0, "")
tf.app.flags.DEFINE_integer("iTrainD", 5, "")

tf.app.flags.DEFINE_float("fLrIniG", learning_rate, "")
tf.app.flags.DEFINE_float("fLrIniD", learning_rate, "")
tf.app.flags.DEFINE_string("oDecay", 'none', "linear, exp, none")

tf.app.flags.DEFINE_float("fBeta1", beta_1, "")
tf.app.flags.DEFINE_float("fBeta2", beta_2, "")
tf.app.flags.DEFINE_float("fEpsilon", epsilon, "")
tf.app.flags.DEFINE_integer("keep_num", keep_num, "")
tf.app.flags.DEFINE_string("pred_g_op", pred_g_op, "max, mean, none")
tf.app.flags.DEFINE_string("optimizer_name", optimizer_name, "adam, sgd, amsgrad adashifts")
tf.app.flags.DEFINE_float("learning_rate", learning_rate, "sys LR")

tf.app.flags.DEFINE_string("oOptG", optimizer_name, "adam, sgd, amsgrad adashift")
tf.app.flags.DEFINE_string("oOptD", optimizer_name, "adam, sgd, amsgrad adashift")

tf.app.flags.DEFINE_float("fWeightLip", 0.1, "")
tf.app.flags.DEFINE_boolean("bMaxGP", True, "")

##################################################### Objectives ##############################################################################################

tf.app.flags.DEFINE_integer("n", 100000000000000000, "")
tf.app.flags.DEFINE_string("sDataSet", "cifar10", "cifar10, mnist")
tf.app.flags.DEFINE_boolean("bLoadCheckpoint", True, "bLoadCheckpoint")

tf.app.flags.DEFINE_boolean("bWGAN", False, "")
tf.app.flags.DEFINE_boolean("bWGANs", True, "")
tf.app.flags.DEFINE_boolean("bLSGAN", False, "")

tf.app.flags.DEFINE_boolean("bLip", True, "")
tf.app.flags.DEFINE_boolean("bGP", False, "")
tf.app.flags.DEFINE_boolean("bLP", False, "")
tf.app.flags.DEFINE_boolean("bCP", False, "")

tf.app.flags.DEFINE_float("fWeightZero", 0.0, "")

################################################# Learning Process ###########################################################################################

tf.app.flags.DEFINE_integer("iMaxIter", iMaxIter, "")
tf.app.flags.DEFINE_integer("iBatchSize", 64, "")

tf.app.flags.DEFINE_boolean("bRampupLr", True, "")
tf.app.flags.DEFINE_boolean("bRampupBeta", False, "")
tf.app.flags.DEFINE_integer("iRampupIter", 0, "")

##################################################### Network Structure #######################################################################################

tf.app.flags.DEFINE_integer("iDimsZ", 100, "")
tf.app.flags.DEFINE_boolean("bSphereZ", True, "")
tf.app.flags.DEFINE_boolean("bUniformZ", False, "")

tf.app.flags.DEFINE_integer("iDimsC", 3, "")
tf.app.flags.DEFINE_boolean("bTanhAtEnd", False, "")

tf.app.flags.DEFINE_integer("iMinSizeG", 4, "")
tf.app.flags.DEFINE_integer("iMinSizeD", 4, "")
tf.app.flags.DEFINE_integer("iMaxSize", 32, "")

tf.app.flags.DEFINE_string("generator", 'generator_block', "")
tf.app.flags.DEFINE_boolean("bBottleNeckG", False, "")
tf.app.flags.DEFINE_integer("iBlockPerLayerG", 2, "")
tf.app.flags.DEFINE_integer("iBaseNumFilterG", 128, "")
tf.app.flags.DEFINE_string("oBlockTypeG", 'res', "dense, res")
tf.app.flags.DEFINE_string("oUpsize", 'upconv', "deconv, upconv")
tf.app.flags.DEFINE_boolean("bActMainPathG", False, "")

tf.app.flags.DEFINE_float("fDimIncreaseRate", 1.0, "")

tf.app.flags.DEFINE_string("discriminator", 'discriminator_block', "")
tf.app.flags.DEFINE_boolean("bBottleNeckD", False, "")
tf.app.flags.DEFINE_integer("iBlockPerLayerD", 2, "")
tf.app.flags.DEFINE_integer("iBaseNumFilterD", 128, "")
tf.app.flags.DEFINE_string("oBlockTypeD", 'res', "dense, res")
tf.app.flags.DEFINE_string("oDownsize", 'downconv', "conv, downconv")
tf.app.flags.DEFINE_boolean("bActMainPathD", False, "")

tf.app.flags.DEFINE_integer("iKsizeG", 3, "")
tf.app.flags.DEFINE_integer("iKsizeD", 3, "")

tf.app.flags.DEFINE_string("oBnG", 'none', "bn, ln, none")
tf.app.flags.DEFINE_string("oBnD", 'none', "bn, ln, none")

tf.app.flags.DEFINE_float("fScaleActG", 0.0, "") #np.sqrt(2/(1+alpha**2))
tf.app.flags.DEFINE_float("fScaleActD", 0.0, "") #np.sqrt(2/(1+alpha**2))
tf.app.flags.DEFINE_string("oActG", 'relu', "elulike, relu, lrelu")
tf.app.flags.DEFINE_string("oActD", 'relu', "elulike, relu, lrelu")

tf.app.flags.DEFINE_boolean("bUseWN", False, "")
tf.app.flags.DEFINE_float("fDefaultGain", 1.00, "")
tf.app.flags.DEFINE_float("fInitWeightStddev", 0.01, "")
tf.app.flags.DEFINE_string("oInitType", 'uniform', "truncated_normal, normal, uniform, orthogonal")

tf.app.flags.DEFINE_integer("GPU", GPU, "")
tf.app.flags.DEFINE_string("sResultDir", "../result/%s/"%T, "where to save the checkpoint and sample")

cfg(sys.argv)     
# cfg()     

GPU_ID = allocate_gpu(cfg.GPU)


np.random.seed(1000)
tf.set_random_seed(1000)

set_enable_bias(True)
set_data_format('NCHW')
set_enable_wn(cfg.bUseWN)
set_default_gain(cfg.fDefaultGain)
set_init_type(cfg.oInitType)
set_init_weight_stddev(cfg.fInitWeightStddev)

c_axis, h_axis, w_axis = [1, 2, 3]

def discriminator_mlp(input, num_logits, name=None):

    layers = []
    iBaseNumFilterD = cfg.iBaseNumFilterD

    with tf.name_scope('' if name is None else name):

        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):

            h0 = input
            layers.append(h0)

            for i in range(cfg.iBlockPerLayerD):

                with tf.variable_scope('layer' + str(i)):

                    h0 = linear(h0, iBaseNumFilterD)
                    layers.append(h0)

                    h0 = normalize(h0, cfg.oBnD)
                    layers.append(h0)

                    h0 = activate(h0, cfg.oActD, cfg.fScaleActD)
                    layers.append(h0)

            h0 = linear(h0, num_logits, name='final_linear')
            layers.append(h0)

        return h0, clear_duplicated_layers(layers)

def discriminator_dcgan(input, num_logits, name):

    layers = []
    iBaseNumFilterD = cfg.iBaseNumFilterD

    with tf.name_scope('' if name is None else name):

        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):

            h0 = input
            layers.append(h0)

            with tf.variable_scope('input'):

                h0 = conv2d(h0, iBaseNumFilterD, ksize=cfg.iKsizeD, stride=1)
                layers.append(h0)

                # h0 = normalize(h0, cfg.oBnD)
                # layers.append(h0)

                h0 = activate(h0, cfg.oActD, cfg.fScaleActD)
                layers.append(h0)

            while True:

                iBaseNumFilterD = int(iBaseNumFilterD * cfg.fDimIncreaseRate)

                with tf.variable_scope('size' + str(h0.get_shape().as_list()[w_axis])):

                    h0 = conv2d(h0, iBaseNumFilterD, ksize=cfg.iKsizeD, stride=2)
                    layers.append(h0)

                    h0 = normalize(h0, cfg.oBnD)
                    layers.append(h0)

                    h0 = activate(h0, cfg.oActD, cfg.fScaleActD)
                    layers.append(h0)

                    if h0.get_shape().as_list()[w_axis] / 2 < cfg.iMinSizeD:
                        break

            with tf.variable_scope('final'):

                h0 = tf.contrib.layers.flatten(h0)
                layers.append(h0)

                h0 = linear(h0, num_logits)
                layers.append(h0)

    return h0, clear_duplicated_layers(layers)


def discriminator_block(input, num_logits, name):

    layers = []
    iBaseNumFilterD = cfg.iBaseNumFilterD

    with tf.name_scope('' if name is None else name):

        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):

            h0 = input
            layers.append(h0)

            with tf.variable_scope('input'):

                h0 = conv2d(h0, iBaseNumFilterD, ksize=cfg.iKsizeD, stride=1)
                layers.append(h0)

            while True:

                with tf.variable_scope('size' + str(h0.get_shape().as_list()[w_axis])):

                    for i in range(cfg.iBlockPerLayerD):

                        with tf.variable_scope('layer' + str(i)):

                            h1 = h0

                            if cfg.bBottleNeckD and h0.get_shape().as_list()[c_axis] > iBaseNumFilterD * 8:

                                with tf.variable_scope('bottleneck'):

                                    h1 = normalize(h1, cfg.oBnD)
                                    layers.append(h1)

                                    h1 = activate(h1, cfg.oActD, cfg.fScaleActD)
                                    layers.append(h1)

                                    h1 = conv2d(h1, iBaseNumFilterD * 4, ksize=1, stride=1)
                                    layers.append(h1)

                            with tf.variable_scope('composite'):

                                h1 = normalize(h1, cfg.oBnD)
                                layers.append(h1)

                                h1 = activate(h1, cfg.oActD, cfg.fScaleActD)
                                layers.append(h1)

                                h1 = conv2d(h1, iBaseNumFilterD, ksize=cfg.iKsizeD, stride=1)
                                layers.append(h1)

                            if cfg.oBlockTypeD == 'dense':

                                h0 = tf.concat(values=[h0, h1], axis=c_axis)

                            elif cfg.oBlockTypeD == 'res':

                                h0 = h0 + h1

                            layers.append(h0)

                if h0.get_shape().as_list()[w_axis] / 2 >= cfg.iMinSizeD:

                    with tf.variable_scope('downsize' + str(h0.get_shape().as_list()[w_axis])):

                        iBaseNumFilterD = int(iBaseNumFilterD * cfg.fDimIncreaseRate)

                        if cfg.bActMainPathD:

                            h0 = normalize(h0, cfg.oBnD)
                            layers.append(h0)

                            h0 = activate(h0, cfg.oActD, cfg.fScaleActD)
                            layers.append(h0)

                        if cfg.oDownsize == 'conv':

                            h0 = conv2d(h0, iBaseNumFilterD, ksize=cfg.iKsizeD, stride=2)
                            layers.append(h0)

                        elif cfg.oDownsize == 'downconv':

                            h0 = avgpool(h0, 2, 2)
                            layers.append(h0)

                            h0 = conv2d(h0, iBaseNumFilterD, ksize=cfg.iKsizeD, stride=1)
                            layers.append(h0)

                        else:

                            h0 = avgpool(h0, 2, 2)
                            layers.append(h0)

                else:
                    break

            with tf.variable_scope('final'):

                if cfg.bActMainPathD:

                    h0 = normalize(h0, cfg.oBnD)
                    layers.append(h0)

                    h0 = activate(h0, cfg.oActD, cfg.fScaleActD)
                    layers.append(h0)

                h0 = tf.contrib.layers.flatten(h0)
                layers.append(h0)

                h0 = linear(h0, num_logits)
                layers.append(h0)

    return h0, clear_duplicated_layers(layers)

def generator_mlp(z=None, name=None):

    layers = []
    iBaseNumFilterG = cfg.iBaseNumFilterG

    with tf.name_scope('' if name is None else name):

        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):

            h0 = z
            layers.append(h0)

            for i in range(cfg.iBlockPerLayerG):

                with tf.variable_scope('layer' + str(i)):

                    h0 = linear(h0, iBaseNumFilterG)
                    layers.append(h0)

                    h0 = normalize(h0, cfg.oBnG)
                    layers.append(h0)

                    h0 = activate(h0, cfg.oActG, cfg.fScaleActG)
                    layers.append(h0)

            h0 = linear(h0, cfg.iDimsC, name='final_linear')
            layers.append(h0)

        return h0, clear_duplicated_layers(layers)

def generator_dcgan(z=None, name=None):

    layers = []

    with tf.name_scope('' if name is None else name):

        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):

            h0 = z
            layers.append(h0)

            size = 32
            iBaseNumFilterG = cfg.iBaseNumFilterG

            while size > cfg.iMinSizeG:

                iBaseNumFilterG = int(iBaseNumFilterG * cfg.fDimIncreaseRate)
                size = size // 2

            with tf.variable_scope('latent'):

                h0 = linear(h0, iBaseNumFilterG * cfg.iMinSizeG * cfg.iMinSizeG)
                layers.append(h0)

                h0 = tf.reshape(h0, [-1, iBaseNumFilterG, cfg.iMinSizeG, cfg.iMinSizeG])
                layers.append(h0)

                h0 = normalize(h0, cfg.oBnG)
                layers.append(h0)

                h0 = activate(h0, cfg.oActG, cfg.fScaleActG)
                layers.append(h0)

            while h0.get_shape().as_list()[w_axis] < 32:

                with tf.variable_scope('size' + str(h0.get_shape().as_list()[w_axis])):

                    iBaseNumFilterG = int(iBaseNumFilterG / cfg.fDimIncreaseRate)

                    h0 = deconv2d(h0, iBaseNumFilterG, ksize=cfg.iKsizeG, stride=2)
                    layers.append(h0)

                    h0 = normalize(h0, cfg.oBnG)
                    layers.append(h0)

                    h0 = activate(h0, cfg.oActG, cfg.fScaleActG)
                    layers.append(h0)

            with tf.variable_scope('final'):

                h0 = deconv2d(h0, cfg.iDimsC, ksize=cfg.iKsizeG, stride=1)
                layers.append(h0)

                if cfg.bTanhAtEnd:
                    h0 = tf.nn.tanh(h0)
                    layers.append(h0)

    return h0, clear_duplicated_layers(layers)

def generator_block(z=None, name=None):

    layers = []

    with tf.name_scope('' if name is None else name):

        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):

            h0 = z
            layers.append(h0)

            size = 32
            iBaseNumFilterG = cfg.iBaseNumFilterG

            while size > cfg.iMinSizeG:
                iBaseNumFilterG = int(iBaseNumFilterG * cfg.fDimIncreaseRate)
                size = size // 2

            with tf.variable_scope('latent'):

                h0 = linear(h0, iBaseNumFilterG * cfg.iMinSizeG * cfg.iMinSizeG)
                layers.append(h0)

                h0 = tf.reshape(h0, [-1, iBaseNumFilterG, cfg.iMinSizeG, cfg.iMinSizeG])
                layers.append(h0)

            while True:

                with tf.variable_scope('size' + str(h0.get_shape().as_list()[w_axis])):

                    for i in range(cfg.iBlockPerLayerG):

                        with tf.variable_scope('layer' + str(i)):

                            h1 = h0

                            if cfg.bBottleNeckG and h0.get_shape().as_list()[c_axis] > iBaseNumFilterG * 8:

                                with tf.variable_scope('bottleneck'):

                                    h1 = normalize(h1, cfg.oBnG)
                                    layers.append(h1)

                                    h1 = activate(h1, cfg.oActG, cfg.fScaleActG)
                                    layers.append(h1)

                                    h1 = conv2d(h1, iBaseNumFilterG * 4, ksize=1, stride=1)
                                    layers.append(h1)

                            with tf.variable_scope('composite'):

                                h1 = normalize(h1, cfg.oBnG)
                                layers.append(h1)

                                h1 = activate(h1, cfg.oActG, cfg.fScaleActG)
                                layers.append(h1)

                                h1 = conv2d(h1, iBaseNumFilterG, ksize=cfg.iKsizeG, stride=1)
                                layers.append(h1)

                            if cfg.oBlockTypeG == 'dense':

                                h0 = tf.concat(values=[h0, h1], axis=c_axis)

                            elif cfg.oBlockTypeG == 'res':

                                h0 = h0 + h1

                            layers.append(h0)

                if h0.get_shape().as_list()[w_axis] < 32:

                    with tf.variable_scope('upsize' + str(h0.get_shape().as_list()[w_axis])):

                        iBaseNumFilterG = int(iBaseNumFilterG / cfg.fDimIncreaseRate)

                        if cfg.bActMainPathG:

                            h0 = normalize(h0, cfg.oBnG)
                            layers.append(h0)

                            h0 = activate(h0, cfg.oActG, cfg.fScaleActG)
                            layers.append(h0)

                        if cfg.oUpsize == 'deconv':

                            h0 = deconv2d(h0, iBaseNumFilterG, ksize=cfg.iKsizeG, stride=2)
                            layers.append(h0)

                        elif cfg.oUpsize == 'upconv':

                            h0 = image_nn_double_size(h0)
                            layers.append(h0)

                            h0 = conv2d(h0, iBaseNumFilterG, ksize=cfg.iKsizeG, stride=1)
                            layers.append(h0)

                        else:

                            h0 = image_nn_double_size(h0)
                            layers.append(h0)

                else:

                    break

            with tf.variable_scope('final'):

                if cfg.bActMainPathG:

                    h0 = normalize(h0, cfg.oBnG)
                    layers.append(h0)

                    h0 = activate(h0, cfg.oActG, cfg.fScaleActG)
                    layers.append(h0)

                h0 = deconv2d(h0, cfg.iDimsC, ksize=cfg.iKsizeG, stride=1)
                layers.append(h0)

                if cfg.bTanhAtEnd:
                    h0 = tf.nn.tanh(h0)
                    layers.append(h0)

    return h0, clear_duplicated_layers(layers)

############################################################################################################################################

def load_dataset(dataset_name):
    if dataset_name == 'cifar10':
        return load_cifar10()
    if dataset_name == 'mnist':
        return load_mnist()

def gen_with_generator():
    while True:
        data = sess.run(fake_datas, feed_dict={z: sample_z(cfg.iBatchSize)})
        yield data

def uniform_gen():
    while True:
        data = np.random.uniform(size=(cfg.iBatchSize,) + np.shape(dataX)[1:], low=-1., high=1.)
        yield data

def param_count(gradient_value):
    total_param_count = 0
    for g, v in gradient_value:
        shape = v.get_shape()
        param_count = 1
        for dim in shape:
            param_count *= int(dim)
        total_param_count += param_count
    return total_param_count

def log_netstate():

    logger.log('\n')
    _gen_layers = sess.run(gen_layers, feed_dict={real_datas: _real_datas, iter_datas: _iter_datas, z: _z})
    for i in range(len(_gen_layers)):
        logger.log('layer values: %8.5f %8.5f    ' % (np.mean(_gen_layers[i]), np.std(_gen_layers[i])) + gen_layers[i].name + ' shape: ' + str(_gen_layers[i].shape))

    logger.log('\n')
    dis_layers = interpolates_layers # + dis_real_layers + dis_fake_layers
    _dis_layers = sess.run(dis_layers, feed_dict={real_datas: _real_datas, iter_datas: _iter_datas, z: _z})
    for i in range(len(_dis_layers)):
        logger.log('layer values: %8.5f %8.5f    ' % (np.mean(_dis_layers[i]), np.std(_dis_layers[i])) + dis_layers[i].name + ' shape: ' + str(_dis_layers[i].shape))

    logger.log('\n')
    _gen_vars, _genvar_tot_gradients = sess.run([gen_vars, genvar_tot_gradients], feed_dict={real_datas: _real_datas, iter_datas: _iter_datas, z: _z})
    for i in range(len(_gen_vars)):
        logger.log('weight values: %8.5f %8.5f, tot gradient: %8.5f %8.5f    ' % (np.mean(_gen_vars[i]), np.std(_gen_vars[i]), np.mean(_genvar_tot_gradients[i]), np.std(_genvar_tot_gradients[i])) + gen_vars[i].name + ' shape: ' + str(gen_vars[i].shape))

    logger.log('\n')
    _dis_vars, _disvar_lip_gradients, _disvar_gan_gradients, _disvar_tot_gradients = sess.run([dis_vars, disvar_lip_gradients, disvar_gan_gradients, disvar_tot_gradients], feed_dict={real_datas: _real_datas, iter_datas: _iter_datas, z: _z})
    for i in range(len(_dis_vars)):
        logger.log('weight values: %8.5f %8.5f, lip gradient: %8.5f %8.5f, gan gradient: %8.5f %8.5f, tot gradient: %8.5f %8.5f    ' % (np.mean(_dis_vars[i]), np.std(_dis_vars[i]), np.mean(_disvar_lip_gradients[i]), np.std(_disvar_lip_gradients[i]), np.mean(_disvar_gan_gradients[i]), np.std(_disvar_gan_gradients[i]), np.mean(_disvar_tot_gradients[i]), np.std(_disvar_tot_gradients[i])) + dis_vars[i].name + ' shape: ' + str(dis_vars[i].shape))

    logger.log('\n')

def sample_z(n):
    if cfg.bUniformZ:
        noise = np.random.rand(n, cfg.iDimsZ)
    elif cfg.bSphereZ:
        noise = np.random.randn(n, cfg.iDimsZ)
        noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    else:
        noise = np.random.randn(n, cfg.iDimsZ)
    return noise

def gen_images(n):
    images = []
    for i in range(n//cfg.iBatchSize+1):
        images.append(sess.run(fake_datas, feed_dict={z: sample_z(cfg.iBatchSize)}))
    images = np.concatenate(images, 0)
    return images[:n]

def gen_images_with_noise(noise):
    images = []
    n = len(noise)
    ii = n // cfg.iBatchSize + 1
    noiseA = sample_z(cfg.iBatchSize * ii)
    noiseA[:n] = noise
    for i in range(ii):
        images.append(sess.run(fake_datas, feed_dict={z: noiseA[cfg.iBatchSize*i:cfg.iBatchSize*(i+1)]}))
    images = np.concatenate(images, 0)
    return images[:n]

ref_icp_preds, ref_icp_activations = None, None
icp_model = PreTrainedInception()

def get_score(samples):

    global ref_icp_preds, ref_icp_activations
    if ref_icp_activations is None:
        logger.log('Evaluating Reference Statistic: icp_model')
        ref_icp_preds, ref_icp_activations = icp_model.get_preds(dataX.transpose(0, 2, 3, 1))
        logger.log('\nref_icp_score: %.3f\n' % InceptionScore.inception_score_H(ref_icp_preds)[0])

    logger.log('Evaluating Generator Statistic')
    icp_preds, icp_activcations = icp_model.get_preds(samples.transpose(0, 2, 3, 1))

    return icp_score, fid

def path2(f, g, n):

    images = []

    s = np.mean((1.0 - f) / g, axis=(1, 2, 3), keepdims=True)

    images.append(f)
    images.append(g / np.max(np.abs(g), axis=(1, 2, 3), keepdims=True))

    for i in range(n):
        ff = f + (i + 1) / n * g * s
        images.append(ff)

    return np.stack(images, 1)

def path(r, f, g, n):

    images = []

    rr = []
    for i in range(len(f)):
        error = np.zeros(len(r))
        for j in range(len(r)):
            # s = np.mean(r[j] - f[i]) / np.mean(g[i])
            # error[j] = np.linalg.norm(r[j] - f[i] - s * g[i])
            g_dir = np.reshape(g[i] / np.linalg.norm(g[i]), [-1])
            rf_dir = np.reshape((r[j]-f[i]) / np.linalg.norm(r[j]-f[i]), [-1])
            error[j] = -g_dir.dot(rf_dir)
        ir = np.argmin(error)
        rr.append(r[ir])
    rr = np.asarray(rr)

    s = np.median((rr-f) / g, axis=(1, 2, 3), keepdims=True)
    # s = np.mean(rr - f, axis=(1, 2, 3), keepdims=True) / np.mean(g, axis=(1, 2, 3), keepdims=True)

    images.append(f)
    images.append(g / np.max(np.abs(g), axis=(1, 2, 3), keepdims=True))

    for i in range(n):
        nn = int(n // 3)
        ff = f + (i+1)/(n-nn) * g * s
        images.append(ff)
        # ff = ff / np.max(np.abs(ff), axis=(1, 2, 3), keepdims=True)
        # images.append(ff)

    images.append(rr)

    return np.stack(images, 1)

def sort_images(images):

    n = len(images)
    flag = np.zeros(n, dtype=int)
    flag[0] = 1

    distance = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            distance[i][j] = np.linalg.norm(images[i]-images[j])

    images2 = [images[0]]

    while len(images2) < n:
        mindisi = 1e20
        mindisi_idx = 0
        for i in np.where(flag == 0)[0]:
            mindisj = 1e20
            for j in np.where(flag == 1)[0]:
                mindisj = min(mindisj, distance[i][j])
            if mindisj < mindisi:
                mindisi = mindisj
                mindisi_idx = i
        images2.append(images[mindisi_idx])
        flag[mindisi_idx] = 1

    images2 = np.concatenate(images2, 0)
    return images2

############################################################################################################################################

sTestName = cfg.sDataSet + ('_' + cfg.sResultTag if len(cfg.sResultTag) else "")
sTestCaseDir = cfg.sResultDir + sTestName + '/'
sSampleDir = sTestCaseDir + '/samples/'
sCheckpointDir = sTestCaseDir + 'checkpoint/'

makedirs(sCheckpointDir)
makedirs(sSampleDir)
makedirs(sTestCaseDir + 'source/code/')
makedirs(sTestCaseDir + 'source/common/')

logger = Logger()
logger.set_dir(sTestCaseDir)
logger.set_casename(sTestName)
logger.log(sTestCaseDir)

commandline = ''
for arg in ['python3'] + sys.argv:
    commandline += arg + ' '
logger.log(commandline)

logger.log(str_flags(cfg.__flags))
logger.log('Using GPU%d\n' % GPU_ID)

copydir(SOURCE_DIR + "common/", sTestCaseDir + 'source/common/')

tf.logging.set_verbosity(tf.logging.ERROR)

############################################################################################################################################

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, intra_op_parallelism_threads=0, inter_op_parallelism_threads=0)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

generator = globals()[cfg.generator]
discriminator = globals()[cfg.discriminator]

dataX, dataY, testX, testY = load_dataset(cfg.sDataSet)
cfg.iDimsC = np.shape(dataX)[1]
cfg.iMaxSize = np.shape(dataX)[2]

real_datas = tf.placeholder(tf.float32, (None,) + np.shape(dataX)[1:], name='real_data')
iter_datas = tf.placeholder(tf.float32, (None,) + np.shape(dataX)[1:], name='iter_data')
# fake_data = tf.placeholder(tf.float32, (None,) + np.shape(r0)[1:], name='fake_data')

z = tf.placeholder(tf.float32, [cfg.iBatchSize, cfg.iDimsZ], name='z')
fake_datas, gen_layers = generator(z, name='fake_data')

real_gen = data_gen_epoch(dataX[:cfg.n], cfg.iBatchSize)
fake_gen = gen_with_generator()

############################################################################################################################################

real_logits, dis_real_layers = discriminator(real_datas, 1, 'real')
fake_logits, dis_fake_layers = discriminator(fake_datas, 1, 'fake')

real_logits = tf.reshape(real_logits, [-1])
fake_logits = tf.reshape(fake_logits, [-1])

if cfg.bWGAN:
    dis_real_loss = -real_logits
    dis_fake_loss = fake_logits
    gen_fake_loss = -fake_logits

elif cfg.bLSGAN:
    dis_real_loss = tf.square(real_logits - 1.0)
    dis_fake_loss = tf.square(fake_logits + 1.0)
    gen_fake_loss = tf.square(fake_logits - 1.0)

elif cfg.bWGANs:
    dis_real_loss = -tf.log_sigmoid(real_logits) - real_logits
    dis_fake_loss = -tf.log_sigmoid(-fake_logits) + fake_logits
    gen_fake_loss = -fake_logits

else:
    dis_real_loss = -tf.log_sigmoid(real_logits)  # tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real_logits))
    dis_fake_loss = -tf.log_sigmoid(-fake_logits)  # tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_logits))
    gen_fake_loss = -tf.log_sigmoid(fake_logits)  # tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones_like(fake_logits))

dis_gan_loss = tf.reduce_mean(dis_fake_loss) + tf.reduce_mean(dis_real_loss)
dis_zero_loss = cfg.fWeightZero * tf.square(tf.reduce_mean(fake_logits) + tf.reduce_mean(real_logits))
dis_tot_loss = dis_gan_loss + dis_zero_loss
gen_tot_loss = gen_gan_loss = tf.reduce_mean(gen_fake_loss)

slopes = tf.constant(0.0)
gradients = tf.constant(0.0)
dis_lip_loss = tf.constant(0.0)
interpolates = tf.constant(0.0)
interpolates_layers = []

if cfg.bLip:

    alpha = tf.random_uniform(shape=[tf.shape(fake_datas)[0], 1, 1, 1], minval=0., maxval=1.)
    differences = fake_datas - real_datas
    interpolates = real_datas + alpha * differences
    if cfg.bMaxGP:
        interpolates = tf.concat([interpolates[:-tf.shape(iter_datas)[0]], iter_datas], 0)

    interpolates_logits, interpolates_layers = discriminator(interpolates, 1, 'inter')
    gradients = tf.gradients(interpolates_logits, interpolates)[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))  # tf.norm()

    if cfg.bMaxGP:
        dis_lip_loss = cfg.fWeightLip * tf.reduce_max(tf.square(slopes))
    elif cfg.bGP:
        dis_lip_loss = cfg.fWeightLip * tf.reduce_mean(tf.square(slopes - 1.0))
    elif cfg.bLP:
        dis_lip_loss = cfg.fWeightLip * tf.reduce_mean(tf.square(tf.maximum(0.0, slopes - 1.0)))
    elif cfg.bCP:
        dis_lip_loss = cfg.fWeightLip * tf.reduce_mean(tf.square(slopes))

    dis_tot_loss += dis_lip_loss

############################################################################################################################################

tot_vars = tf.trainable_variables()
dis_vars = [var for var in tot_vars if 'discriminator' in var.name]
gen_vars = [var for var in tot_vars if 'generator' in var.name]

global_step = tf.Variable(0, trainable=False, name='global_step')
step_op = tf.assign_add(global_step, 1)

if cfg.bRampupBeta:
    beta1 = rampup(global_step, cfg.iRampupIter) * tf.constant(cfg.fBeta1)
    beta2 = (0.9 + rampup(global_step, cfg.iRampupIter) * tf.constant(cfg.fBeta2-0.9)) if cfg.fBeta2 > 0.9 else cfg.fBeta2
else:
    beta1 = tf.constant(cfg.fBeta1)
    beta2 = tf.constant(cfg.fBeta2)

gen_lr = tf.constant(cfg.fLrIniG) * rampup(global_step, cfg.iRampupIter) if cfg.bRampupLr else tf.constant(cfg.fLrIniG)
if 'linear' in cfg.oDecay:
    gen_lr = gen_lr * tf.maximum(0., 1. - (tf.cast(global_step, tf.float32) / cfg.iMaxIter))
elif 'exp' in cfg.oDecay:
    gen_lr = tf.train.exponential_decay(gen_lr, global_step, cfg.iMaxIter // 5, 0.25, True)

gen_optimizer = None
if cfg.oOptG == 'sgd':
    gen_optimizer = optimizer_shift.Grad(learning_rate=gen_lr)
elif cfg.oOptG == 'adam':
    gen_optimizer = optimizer_shift.Adam(learning_rate=gen_lr, beta1=beta1, beta2=beta2, epsilon=cfg.fEpsilon)
elif cfg.oOptG == 'max':
    gen_optimizer = optimizer_shift.AdamMax(learning_rate=gen_lr, beta1=beta1, beta2=beta2, epsilon=cfg.fEpsilon)
elif cfg.oOptG == 'adashift':
    gen_optimizer = optimizer_shift.AdaShift(learning_rate=gen_lr, pred_g_op=cfg.pred_g_op, keep_num=cfg.keep_num, beta1=cfg.fBeta1, beta2=beta2, epsilon=cfg.fEpsilon)
elif cfg.oOptG == 'amsgrad':
    gen_optimizer = optimizer_shift.AMSGrad(learning_rate=gen_lr, beta1=beta1,beta2=beta2,epsilon=cfg.fEpsilon)




gen_gradient_values = gen_optimizer.compute_gradients(gen_tot_loss, var_list=gen_vars)
gen_optimize_ops = gen_optimizer.apply_gradients(gen_gradient_values)

dis_lr = tf.constant(cfg.fLrIniD) * rampup(global_step, cfg.iRampupIter) if cfg.bRampupLr else tf.constant(cfg.fLrIniD)
if 'linear' in cfg.oDecay:
    dis_lr = dis_lr * tf.maximum(0., 1. - (tf.cast(global_step, tf.float32) / cfg.iMaxIter))
elif 'exp' in cfg.oDecay:
    dis_lr = tf.train.exponential_decay(dis_lr, global_step, cfg.iMaxIter // 5, 0.25, True)

dis_optimizer = None
if cfg.oOptD == 'sgd':
    dis_optimizer = optimizer_shift.Grad(learning_rate=dis_lr)
elif cfg.oOptD == 'adam':
    dis_optimizer = optimizer_shift.Adam(learning_rate=dis_lr, beta1=beta1, beta2=beta2, epsilon=cfg.fEpsilon)
elif cfg.oOptD == 'amsgrad':
    dis_optimizer = optimizer_shift.AMSGrad(learning_rate=dis_lr, beta1=beta1, beta2=beta2, epsilon=cfg.fEpsilon)
elif cfg.oOptD == 'max':
    dis_optimizer = optimizer_shift.AdamMax(learning_rate=dis_lr, beta1=beta1, beta2=beta2, epsilon=cfg.fEpsilon)
elif cfg.oOptD == 'adashift':
    dis_optimizer = optimizer_shift.AdaShift(learning_rate=dis_lr, keep_num=cfg.keep_num, beta1=cfg.fBeta1, beta2=beta2, pred_g_op=cfg.pred_g_op, epsilon=cfg.fEpsilon)

dis_gradient_values = dis_optimizer.compute_gradients(dis_tot_loss, var_list=dis_vars)
dis_optimize_ops = dis_optimizer.apply_gradients(dis_gradient_values)

############################################################################################################################################

real_gradients = tf.gradients(real_logits, real_datas)[0]
fake_gradients = tf.gradients(fake_logits, fake_datas)[0]

varphi_gradients = tf.gradients(dis_real_loss, real_logits)[0]
phi_gradients = tf.gradients(dis_fake_loss, fake_logits)[0]

disvar_lip_gradients = tf.gradients(dis_lip_loss, dis_vars)
disvar_gan_gradients = tf.gradients(dis_gan_loss, dis_vars)
disvar_tot_gradients = tf.gradients(dis_tot_loss, dis_vars)
genvar_tot_gradients = tf.gradients(gen_tot_loss, gen_vars)

disvar_lip_gradients = [tf.constant(0.0) if grad is None else grad for grad in disvar_lip_gradients]

saver = tf.train.Saver(max_to_keep=1000)
writer = tf.summary.FileWriter(sTestCaseDir, sess.graph)

############################################################################################################################################

iter = 0
last_save_time = last_icp_time = last_log_time = last_plot_time = time.time()

if cfg.bLoadCheckpoint:
    try:
        if load_model(saver, sess, sCheckpointDir):
            logger.log(" [*] Load SUCCESS")
            iter = sess.run(global_step)
            logger.load()
            logger.tick(iter)
            logger.log('\n')
            logger.flush()
            logger.log('\n')
            logger.plot()
        else:
            assert False
    except:
        logger.clear()
        logger.log(" [*] Load FAILED")
        ini_model(sess)
else:
    ini_model(sess)

alphat = np.random.uniform(size=[cfg.iBatchSize, 1, 1, 1])
_real_datas = real_gen.__next__()
_fake_datas = fake_gen.__next__()
_iter_datas = (_real_datas * alphat + _fake_datas * (1-alphat))[:cfg.iBatchSize // 2]
_z = sample_z(cfg.iBatchSize)

log_netstate()
logger.log("Generator Total Parameter Count: {}".format(locale.format("%d", param_count(gen_gradient_values), grouping=True)))
logger.log("Discriminator Total Parameter Count: {}\n".format(locale.format("%d", param_count(dis_gradient_values), grouping=True)))

fixed_noise = sample_z(256)
start_time = time.time()

while iter < cfg.iMaxIter:

    iter += 1
    train_start_time = time.time()

    for i in range(cfg.iTrainD):
        _, _dis_tot_loss, _dis_gan_loss, _dis_lip_loss, _interpolates, _dphi, _dvarphi, _slopes, _dis_zero_loss, _dis_lr, _beta1, _beta2, _real_logits, _fake_logits = sess.run(
            [dis_optimize_ops, dis_tot_loss, dis_gan_loss, dis_lip_loss, interpolates, phi_gradients, varphi_gradients, slopes, dis_zero_loss, dis_lr, beta1, beta2, real_logits, fake_logits],
            feed_dict={real_datas: real_gen.__next__(), iter_datas: _iter_datas, z: sample_z(cfg.iBatchSize)})
        print('+Task: %s in %d'%(T,GPU_ID))

    for i in range(cfg.iTrainG):
        _, _gen_total_loss, _gen_gan_loss, _gen_lr = sess.run([gen_optimize_ops, gen_tot_loss, gen_gan_loss, gen_lr],
                                                              feed_dict={z: sample_z(cfg.iBatchSize)})

    sess.run(step_op)
    logger.info('time_train', time.time() - train_start_time)
    logger.info('time_hour_remain', (time.time() - start_time) / iter * (cfg.iMaxIter - iter) / 3600)

    log_start_time = time.time()

    logger.tick(iter)

    if cfg.iTrainD > 0:
        logger.info('klrD', _dis_lr*1000)
        logger.info('beta1', _beta1)
        logger.info('beta2', _beta2)

        logger.info('logit_real', np.mean(_real_logits))
        logger.info('logit_fake', np.mean(_fake_logits))

        logger.info('loss_dis_gp', _dis_lip_loss)
        logger.info('loss_dis_zero', _dis_zero_loss)

        logger.info('loss_dis_gan', _dis_gan_loss)
        logger.info('loss_dis_tot', _dis_tot_loss)

        logger.info('d_phi', np.mean(_dphi))
        logger.info('d_varphi', np.mean(_dvarphi))

        logger.info('slopes_max', np.max(_slopes))
        logger.info('slopes_mean', np.mean(_slopes))

    if cfg.iTrainG > 0:
        logger.info('klrG', _gen_lr*1000)
        logger.info('loss_gen_gan', _gen_gan_loss)
        logger.info('loss_gen_tot', _gen_total_loss)


    if cfg.bLip and cfg.bMaxGP:
        _iter_datas = _interpolates[np.argsort(-np.asarray(_slopes))[:len(_iter_datas)]]

    if np.any(np.isnan(_real_logits)) or np.any(np.isnan(_fake_logits)):
        log_netstate()
        logger.flush()
        exit()

    if time.time() - last_icp_time > 60*60 and cfg.sDataSet == 'cifar10' and cfg.n > 1000:
        try:
            icp_score, fid = get_score(gen_images(50000))
            logger.info('score_icp', icp_score)
            logger.info('score_fid', fid)
        except:
            pass
        last_icp_time = time.time()

    if time.time() - last_save_time > 60*30:
        logger.save()
        save_model(saver, sess, sCheckpointDir, step=iter)
        last_save_time = time.time()

    if time.time() - last_plot_time > 60*10:
        logger.plot()
        log_netstate()

        # f0 = gen_images(25)
        # grad_path = path2(f0, sess.run(real_gradients, feed_dict={real_datas: f0}), 14)
        # save_images(grad_path.reshape(-1, cfg.iDimsC, 32, 32).transpose([0, 2, 3, 1]), [np.shape(grad_path)[0], np.shape(grad_path)[1]], sSampleDir + 'grad_path25x14_%d.png' % iter)

        f0 = gen_images(128)
        g0 = sess.run(real_gradients, feed_dict={real_datas: f0})
        g0 = g0 / np.max(np.abs(g0), axis=(1, 2, 3), keepdims=True)
        grad_image = np.stack([f0, g0], 1)
        save_images(grad_image.reshape(-1, cfg.iDimsC, 32, 32).transpose([0, 2, 3, 1]), [16, 16], sSampleDir + 'grad_image16x16_%d.png' % iter)

        f0 = sort_images(gen_images(256))
        save_images(f0.reshape(-1, cfg.iDimsC, 32, 32).transpose([0, 2, 3, 1]), [16, 16], sSampleDir + 'gen_image16x16_%d.png' % iter)

        last_plot_time = time.time()

    if time.time() - last_log_time > 60*1:
        _fixed_noise_gen = gen_images_with_noise(fixed_noise)[:256]
        save_images(_fixed_noise_gen.transpose(0, 2, 3, 1), [16, 16], '{}/train_{:02d}_{:04d}.png'.format(sSampleDir, iter // 10000, iter % 10000))

        logger.info('time_log', time.time() - log_start_time)
        logger.flush()

        last_log_time = time.time()
