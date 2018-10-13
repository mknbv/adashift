import warnings
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import linalg_ops

__data_format__ = "NCHW"
__enable_bias__ = True

__enable_wn__ = False
__scale_weight_after_init__ = True

__default_gain__ = 1.0
__init_weight_stddev__ = 1.00  # small __init_weight_stddev__, such as 0.01, along with Adam, seems benefit the generalization.
__init_distribution_type__ = 'uniform'  # truncated_normal, normal, uniform, orthogonal


def set_data_format(data_format):
    global __data_format__
    __data_format__ = data_format

def set_enable_bias(boolvalue):
    global __enable_bias__
    __enable_bias__ = boolvalue

def set_enable_wn(boolvalue):
    global __enable_wn__
    __enable_wn__ = boolvalue

def set_init_weight_stddev(stddev):
    global __init_weight_stddev__
    __init_weight_stddev__ = stddev

def set_default_gain(stddev):
    global __default_gain__
    __default_gain__ = stddev

def set_init_type(type_str):
    global __init_distribution_type__
    __init_distribution_type__ = type_str

def set_scale_weight_after_init(boolvalue):
    global __scale_weight_after_init__
    __scale_weight_after_init__ = boolvalue


def identity(inputs):
    return inputs


def cond_batch_norm(axes, inputs, labels=None, n_labels=None, name='cond_bn'):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        mean, var = tf.nn.moments(inputs, axes, keep_dims=True)
        n_neurons = inputs.get_shape().as_list()[list((set(range(len(inputs.get_shape()))) - set(axes)))[0]]

        if labels is None:

            scale = tf.get_variable('scale', initializer=np.ones(var.get_shape().as_list(), dtype='float32'))
            offset = tf.get_variable('offset', initializer=np.zeros(mean.get_shape().as_list(), dtype='float32'))
            result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-5)

        else:

            scale_m = tf.get_variable('scale', initializer=np.ones([n_labels, n_neurons], dtype='float32'))
            offset_m = tf.get_variable('offset', initializer=np.zeros([n_labels, n_neurons], dtype='float32'))
            scale = tf.nn.embedding_lookup(scale_m, labels, name='embedding_scale')
            offset = tf.nn.embedding_lookup(offset_m, labels, name='embedding_offset')
            result = tf.nn.batch_normalization(inputs, mean, var, offset[:, :, None, None], scale[:, :, None, None], 1e-5)

    return result


def norm(input, reduce_axis, parm_axis, bOffset=True, bScale=True, epsilon=0.001, name='norm'):

    assert not __enable_sn__

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        input_shape = input.get_shape().as_list()

        params_shape = [1] * len(input_shape)
        for i in parm_axis:
            params_shape[i] = input_shape[i]

        offset, scale = None, None
        if bScale:
            scale = tf.get_variable('scale', shape=params_shape, initializer=tf.ones_initializer())
        if bOffset:
            offset = tf.get_variable('offset', shape=params_shape, initializer=tf.zeros_initializer())

        mean, variance = tf.nn.moments(input, reduce_axis, keep_dims=True)

        # return (input - mean) * tf.rsqrt(variance + epsilon) * scale + offset
        outputs = tf.nn.batch_normalization(input, mean, variance, offset, scale, epsilon)

    return outputs


def batch_norm(input, bOffset=True, bScale=True, epsilon=0.001, name='bn'):

    assert not __enable_sn__

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        input_shape = input.get_shape().as_list()

        if len(input_shape) == 4:

            h_axis, w_axis, c_axis = [1, 2, 3] if __data_format__ == "NHWC" else [2, 3, 1]
            reduce_axis = [0, h_axis, w_axis]
            params_shape = [1 for i in range(len(input_shape))]
            params_shape[c_axis] = input_shape[c_axis]

            offset, scale = None, None
            if bScale:
                scale = tf.get_variable('scale', shape=params_shape, initializer=tf.ones_initializer())
            if bOffset:
                offset = tf.get_variable('offset', shape=params_shape, initializer=tf.zeros_initializer())

            batch_mean, batch_variance = tf.nn.moments(input, reduce_axis, keep_dims=True)
            outputs = tf.nn.batch_normalization(input, batch_mean, batch_variance, offset, scale, epsilon)
        else:

            assert len(input.get_shape()) == 2

            axis = [0]
            params_shape = input.get_shape()[1]

            offset, scale = None, None
            if bScale:
                scale = tf.get_variable('scale', shape=params_shape, initializer=tf.ones_initializer())
            if bOffset:
                offset = tf.get_variable('offset', shape=params_shape, initializer=tf.zeros_initializer())

            batch_mean, batch_variance = tf.nn.moments(input, axis)
            outputs = tf.nn.batch_normalization(input, batch_mean, batch_variance, offset, scale, epsilon)

    # Note: here we did not do the moving average (for testing). which we usually not use.

    return outputs


def deconv2d(input, output_dim, ksize=3, stride=1, padding='SAME', bBias=False, name='deconv2d', gain=None):

    def get_deconv_dim(spatial_size, stride_size, kernel_size, padding):
        spatial_size *= stride_size
        if padding == 'VALID':
            spatial_size += max(kernel_size - stride_size, 0)
        return spatial_size

    input_shape = input.get_shape().as_list()
    h_axis, w_axis, c_axis = [1, 2, 3] if __data_format__ == "NHWC" else [2, 3, 1]
    strides = [1, stride, stride, 1] if __data_format__ == "NHWC" else [1, 1, stride, stride]

    output_shape = list(input_shape)
    output_shape[h_axis] = get_deconv_dim(input_shape[h_axis], stride, ksize, padding)
    output_shape[w_axis] = get_deconv_dim(input_shape[w_axis], stride, ksize, padding)
    output_shape[c_axis] = output_dim

    gain = __default_gain__ if gain is None else gain

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        if __enable_wn__:
            w = tf.get_variable('w', [ksize, ksize, output_dim, input_shape[c_axis]], initializer=initializer(__init_distribution_type__, [0, 1, 3], __init_weight_stddev__))
            g = tf.get_variable('g', initializer=tf.ones_like(tf.reduce_sum(w, [0, 1, 3], keepdims=True)) * stride * gain)
            w = g * tf.nn.l2_normalize(w, [0, 1, 3])
        elif __scale_weight_after_init__:
            scale = gain / np.sqrt(float(ksize * ksize * input_shape[c_axis])) / __init_weight_stddev__ * stride
            w = scale * tf.get_variable('w', [ksize, ksize, output_dim, input_shape[c_axis]], initializer=initializer(__init_distribution_type__, [0, 1, 3], __init_weight_stddev__))
        else:
            scale = gain / np.sqrt(float(ksize * ksize * input_shape[c_axis])) / __init_weight_stddev__ * stride
            w = tf.get_variable('w', [ksize, ksize, output_dim, input_shape[c_axis]], initializer=initializer(__init_distribution_type__, [0, 1, 3], __init_weight_stddev__ * scale))

        if __enable_sn__:
            w = spectral_normed_weight(w)[0]

        x = tf.nn.conv2d_transpose(input, w, output_shape=output_shape, strides=strides, padding=padding, data_format=__data_format__)

        if __enable_bias__ or bBias:
            b = tf.get_variable('b', initializer=tf.constant_initializer(0.0), shape=[output_dim])
            x = tf.nn.bias_add(x, b, data_format=__data_format__)

    return x


def conv2d(input, output_dim, ksize=3, stride=1, padding='SAME', bBias=False, name='conv2d', gain=None):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        gain = __default_gain__ if gain is None else gain

        input_shape = input.get_shape().as_list()
        h_axis, w_axis, c_axis = [1, 2, 3] if __data_format__ == "NHWC" else [2, 3, 1]
        strides = [1, stride, stride, 1] if __data_format__ == "NHWC" else [1, 1, stride, stride]

        if __enable_wn__:
            w = tf.get_variable('w', [ksize, ksize, input_shape[c_axis], output_dim], initializer=initializer(__init_distribution_type__, [0, 1, 2], __init_weight_stddev__))
            g = tf.get_variable('g', initializer=tf.ones_like(tf.reduce_sum(w, [0, 1, 2], keepdims=True)) * gain)
            w = g * tf.nn.l2_normalize(w, [0, 1, 2])
        elif __scale_weight_after_init__:
            scale = gain / np.sqrt(float(ksize * ksize * input_shape[c_axis])) / __init_weight_stddev__
            w = tf.get_variable('w', [ksize, ksize, input_shape[c_axis], output_dim], initializer=initializer(__init_distribution_type__, [0, 1, 2], __init_weight_stddev__)) * scale
        else:
            scale = gain / np.sqrt(float(ksize * ksize * input_shape[c_axis])) / __init_weight_stddev__
            w = tf.get_variable('w', [ksize, ksize, input_shape[c_axis], output_dim], initializer=initializer(__init_distribution_type__, [0, 1, 2], __init_weight_stddev__ * scale))

        if __enable_sn__:
            w = spectral_normed_weight(w)[0]

        x = tf.nn.conv2d(input, w, strides=strides, padding=padding, data_format=__data_format__)

        if __enable_bias__ or bBias:
            b = tf.get_variable('b', initializer=tf.constant_initializer(0.0), shape=[output_dim])
            x = tf.nn.bias_add(x, b, data_format=__data_format__)

    return x


def linear(input, output_size, bBias=False, name='linear', gain=None):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        gain = __default_gain__ if gain is None else gain

        if len(input.get_shape().as_list()) > 2:
            warnings.warn('using ops \'linear\' with input shape' + str(input.get_shape().as_list()))
            input = tf.reshape(input, [input.get_shape().as_list()[0], -1])

        if __enable_wn__:
            w = tf.get_variable('w', [input.get_shape().as_list()[1], output_size], initializer=initializer(__init_distribution_type__, [0], __init_weight_stddev__))
            g = tf.get_variable('g', initializer=tf.ones_like(tf.reduce_sum(tf.square(w), [0], keepdims=True)) * gain)
            w = g * tf.nn.l2_normalize(w, [0])
        elif __scale_weight_after_init__:
            scale = gain / np.sqrt(float(input.get_shape().as_list()[1])) / __init_weight_stddev__
            w = tf.get_variable('w', [input.get_shape().as_list()[1], output_size], initializer=initializer(__init_distribution_type__, [0], __init_weight_stddev__)) * scale
        else:
            scale = gain / np.sqrt(float(input.get_shape().as_list()[1])) / __init_weight_stddev__
            w = tf.get_variable('w', [input.get_shape().as_list()[1], output_size], initializer=initializer(__init_distribution_type__, [0], __init_weight_stddev__ * scale))

        if __enable_sn__:
            w = spectral_normed_weight(w)[0]

        x = tf.matmul(input, w)

        if __enable_bias__ or bBias:
            b = tf.get_variable('b', initializer=tf.constant_initializer(0.0), shape=[output_size])
            x = tf.nn.bias_add(x, b)

    return x


def avgpool(input, ksize, stride, name='avgpool', scale_up=False):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        kernel = [1, ksize, ksize, 1] if __data_format__ == "NHWC" else [1, 1, ksize, ksize]
        strides = [1, stride, stride, 1] if __data_format__ == "NHWC" else [1, 1, stride, stride]

        input = tf.nn.avg_pool(input, ksize=kernel, strides=strides, padding='VALID', name=name, data_format=__data_format__)

        if scale_up:
            input *= ksize

    return input


def maxpool(input, ksize, stride, name='maxpool'):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        kernel = [1, ksize, ksize, 1] if __data_format__ == "NHWC" else [1, 1, ksize, ksize]
        strides = [1, stride, stride, 1] if __data_format__ == "NHWC" else [1, 1, stride, stride]

        input = tf.nn.max_pool(input, ksize=kernel, strides=strides, padding='VALID', name=name, data_format=__data_format__)

    return input


def image_nn_double_size(input, name='resize'):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        h_axis, w_axis, c_axis = [1, 2, 3] if __data_format__ == "NHWC" else [2, 3, 1]
        input = tf.concat([input, input, input, input], axis=c_axis)
        input = tf.depth_to_space(input, 2, data_format=__data_format__)

    return input


def noise(input, stddev, bAdd=False, bMulti=True, keep_prob=None, name='noise'):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        if bAdd:
            input = input + tf.truncated_normal(tf.shape(input), 0, stddev, name=name)

        if bMulti:
            if keep_prob is not None:
                stddev = np.sqrt((1 - keep_prob) / keep_prob)  # get 'equivalent' stddev to dropout of keep_prob
            input = input * tf.truncated_normal(tf.shape(input), 1, stddev, name=name)

    return input


def dropout(input, drop_prob, name='dropout'):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        keep_prob = 1.0 - drop_prob
        # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(tf.shape(input))
        binary_tensor = tf.floor(random_tensor)
        input = input * binary_tensor * np.sqrt(1.0 / keep_prob)
        # input = tf.nn.dropout(input, 1.0 - drop_prob, name=name) * (1.0 - drop_prob)

    return input


def normalize(input, oBn, name=None):

    name = oBn if name is None else name

    if oBn == 'bn':
        return batch_norm(input, name=name)
    elif oBn == 'pn':
        assert  __data_format__ == 'NCHW'
        with tf.variable_scope(name):
            return input * tf.rsqrt(tf.reduce_mean(tf.square(input), axis=1, keepdims=True) + 1e-8)
    elif oBn == 'ln':
        if __data_format__ == 'NCHW':
            return norm(input, list(np.asarray(range(len(input.get_shape())-1))+1), [1], name=name)
        else:
            return norm(input, list(np.asarray(range(len(input.get_shape())-1))+1), [min(len(input.get_shape())-1, 3)], name=name)
    else:
        assert oBn == 'none'
        return input


def activate(input, oAct, scale=1.0, name=None):

    with tf.variable_scope(oAct if name is None else name):

        default_scale = 1.0

        if oAct == 'elu':
            input = tf.nn.elu(input)

        elif oAct == 'elulike':
            input = tf.nn.softplus(input) - np.log(2.0)
            default_scale = 2.0

        elif oAct == 'relu':
            input = tf.nn.relu(input)
            default_scale = 1.414

        elif oAct == 'lrelu':
            input = tf.nn.leaky_relu(input)
            default_scale = 1.38675

        elif oAct == 'softmax':
            input = tf.nn.softmax(input)

        elif oAct == 'tanh':
            input = tf.nn.tanh(input)

        elif oAct == 'crelu':
            input = tf.nn.crelu(input)

        elif oAct == 'selu':
            input = tf.nn.selu(input)

        elif oAct == 'swish':
            input = tf.nn.sigmoid(input) * input

        elif oAct == 'softplus':
            input = tf.nn.softplus(input)

        elif oAct == 'softsign':
            input = tf.nn.softsign(input)

        elif oAct == 'pwxb':
            shape = input.get_shape()

            if len(shape) == 4:
                if __data_format__ == 'NCHW':
                    inputA = input[:, :shape[1] // 2, :, :]
                    inputB = input[:, shape[1] // 2:, :, :]
                else:
                    inputA = input[:, :, :, :shape[3] // 2]
                    inputB = input[:, :, :, shape[3] // 2:]
            else:
                inputA = input[:, :shape[1] // 2]
                inputB = input[:, shape[1] // 2:]

            input = inputA * inputB

        elif oAct == 'twxb':
            shape = input.get_shape()

            if len(shape) == 4:
                if __data_format__ == 'NCHW':
                    inputA = input[:, :shape[1] // 2, :, :]
                    inputB = input[:, shape[1] // 2:, :, :]
                else:
                    inputA = input[:, :, :, :shape[3] // 2]
                    inputB = input[:, :, :, shape[3] // 2:]
            else:
                inputA = input[:, :shape[1] // 2]
                inputB = input[:, shape[1] // 2:]

            input = tf.nn.softplus(inputA) * inputB

            # xx = tf.nn.selu(tf.random_normal([1000000]))
            # yy = tf.nn.selu(tf.random_normal([1000000]))
            # print(tf.Session().run([tf.nn.moments(xx, 0), tf.nn.moments(tf.nn.softplus(xx) * yy, 0)]))

        else:
            assert oAct == 'none'

        if scale == 0.0:
            assert oAct in ['elulike', 'relu', 'lrelu']
            scale = default_scale

        input *= scale

    return input


def lnoise(input, fNoise, fDrop):

    if fNoise > 0:
        input = noise(input=input, stddev=fNoise, bMulti=True, bAdd=False)

    if fDrop > 0:
        input = dropout(input=input, drop_prob=fDrop)

    return input


def minibatch_feature(input, n_kernels=100, dim_per_kernel=5, name='minibatch'):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        if len(input.get_shape()) > 2:
            input = tf.reshape(input, [input.get_shape().as_list()[0], -1])

        batchsize = input.get_shape().as_list()[0]

        x = linear(input, n_kernels * dim_per_kernel)
        x = tf.reshape(x, [-1, n_kernels, dim_per_kernel])

        mask = np.zeros([batchsize, batchsize])
        mask += np.eye(batchsize)
        mask = np.expand_dims(mask, 1)
        mask = 1. - mask
        rscale = 1.0 / np.sum(mask)

        abs_dif = tf.reduce_sum(tf.abs(tf.expand_dims(x, 3) - tf.expand_dims(tf.transpose(x, [1, 2, 0]), 0)), 2)
        masked = tf.exp(-abs_dif) * mask
        dist = tf.reduce_sum(masked, 2) * rscale

    return dist


def channel_concat(x, y):

    x_shapes = x.get_shape().as_list()
    y_shapes = y.get_shape().as_list()
    assert y_shapes[0] == x_shapes[0]

    y = tf.reshape(y, [y_shapes[0], 1, 1, y_shapes[1]]) * tf.ones([y_shapes[0], x_shapes[1], x_shapes[2], y_shapes[1]])
    return tf.concat([x, y], 3)


def normalized_orthogonal_initializer(flatten_axis, stddev=1.0):

    def _initializer(shape, dtype=None, partition_info=None):

        if len(shape) < 2:
            raise ValueError("The tensor to initialize must be at least two-dimensional")

        num_rows = 1
        for dim in [shape[i] for i in flatten_axis]:
            num_rows *= dim
        num_cols = shape[list(set(range(len(shape))) - set(flatten_axis))[0]]

        flat_shape = (num_cols, num_rows) if num_rows < num_cols else (num_rows, num_cols)

        a = random(flat_shape, type='uniform', stddev=1.0)
        q, r = linalg_ops.qr(a, full_matrices=False)

        q *= np.sqrt(flat_shape[0])

        if num_rows < num_cols:
            q = tf.matrix_transpose(q)

        return stddev * tf.reshape(q, shape)

    return _initializer


def initializer(type, flatten_axis, stddev):

    if type == 'normal':
        return tf.random_normal_initializer(stddev=stddev)

    elif type == 'uniform':
        return tf.random_uniform_initializer(minval=-stddev * np.sqrt(3.0), maxval=stddev * np.sqrt(3.0))

    elif type == 'truncated_normal':
        return tf.truncated_normal_initializer(stddev=stddev * np.sqrt(1.3))

    elif type == 'orthogonal':
        return normalized_orthogonal_initializer(flatten_axis, stddev=stddev)


def random(shape, type, stddev):

    if type == 'normal':
        return tf.random_normal(shape, stddev=stddev)

    elif type == 'uniform':
        return tf.random_uniform(shape, minval=-stddev * np.sqrt(3.0), maxval=stddev * np.sqrt(3.0))

    elif type == 'truncated_normal':
        return tf.truncated_normal(shape, stddev=stddev * np.sqrt(1.3))


__enable_sn__ = False
__enable_snk__ = False

SPECTRAL_NORM_K_LIST = []
SPECTRAL_NORM_UV_LIST = []
SPECTRAL_NORM_SIGMA_LIST = []

SPECTRAL_NORM_K_INIT_OPS_LIST = []
SPECTRAL_NORM_UV_UPDATE_OPS_LIST = []


def set_enable_sn(value):
    global __enable_sn__
    __enable_sn__ = value


def set_enable_snk(value):
    global __enable_snk__
    __enable_snk__ = value


def spectral_normed_weight(W, num_iters=10, bUseCollection=True):

    def _l2normalize(v, eps=1e-12):
        return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

    W_shape = W.shape.as_list()
    W_reshaped = tf.reshape(W, [-1, W_shape[-1]])

    u = tf.get_variable('u', [1, W_reshaped.shape.as_list()[1]], initializer=tf.truncated_normal_initializer(), trainable=False)
    v = tf.get_variable('v', [1, W_reshaped.shape.as_list()[0]], initializer=tf.truncated_normal_initializer(), trainable=False)

    def power_iteration(i, u_i, v_i):
        v_ip1 = _l2normalize(tf.matmul(u_i, tf.transpose(W_reshaped)))
        u_ip1 = _l2normalize(tf.matmul(v_ip1, W_reshaped))
        return i + 1, u_ip1, v_ip1

    _, u_final, v_final = tf.while_loop(
        cond=lambda i, _1, _2: i < num_iters,
        body=power_iteration,
        loop_vars=(tf.constant(0, dtype=tf.int32), u, v)
    )

    if bUseCollection:
        if u.name not in SPECTRAL_NORM_UV_LIST:
            SPECTRAL_NORM_UV_LIST.append(u.name)
            SPECTRAL_NORM_UV_UPDATE_OPS_LIST.append(u.assign(u_final))
        if v.name not in SPECTRAL_NORM_UV_LIST:
            SPECTRAL_NORM_UV_LIST.append(v.name)
            SPECTRAL_NORM_UV_UPDATE_OPS_LIST.append(v.assign(v_final))
        sigma = tf.matmul(tf.matmul(v, W_reshaped), tf.transpose(u))[0, 0]
    else:
        with tf.control_dependencies([u.assign(u_final), v.assign(v_final)]):
            sigma = tf.matmul(tf.matmul(v, W_reshaped), tf.transpose(u))[0, 0]

    if __enable_snk__:
        k = tf.get_variable('k', initializer=sigma, trainable=True)
        if k.name not in [var.name for var in SPECTRAL_NORM_K_LIST]:
            SPECTRAL_NORM_K_LIST.append(k)
            SPECTRAL_NORM_SIGMA_LIST.append(sigma)
            SPECTRAL_NORM_K_INIT_OPS_LIST.append(k.assign(sigma))
        W_bar = W_reshaped / sigma * k
    else:
        W_bar = W_reshaped / sigma

    W_bar = tf.reshape(W_bar, W_shape)

    return W_bar, sigma, u, v