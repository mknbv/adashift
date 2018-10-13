import os
from shutil import *
import random, math
import scipy.misc
import numpy as np
import tensorflow as tf

def clear_duplicated_layers(layers):
    layers0 = [layers[0]]
    for layer in layers:
        if layer.name != layers0[-1].name:
            layers0.append(layer)
    return layers0

def allocate_gpu(gpu_id=-1, maxLoad=0.1, maxMem=0.5, order='memory'):
    if gpu_id == -1:
        try:
            import common.GPUtil as GPUtil
            gpu_id = GPUtil.getFirstAvailable(order=order, maxLoad=maxLoad, maxMemory=maxMem)[0]
        except:
            gpu_id = 0
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return gpu_id


def ini_model(sess):
    sess.run(tf.global_variables_initializer())


def save_model(saver, sess, checkpoint_dir, step=None):
    makedirs(checkpoint_dir)
    model_name = "model"
    saver.save(sess, checkpoint_dir + model_name, global_step=step)


def load_model(saver, sess, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        return True
    else:
        return False


from functools import reduce
import operator


def prod(iterable):
    return reduce(operator.mul, iterable, 1)

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def mean(x):
    try:
        return np.mean(x).__float__()
    except:
        return 0.

def std(x):
    try:
        return np.std(x).__float__()
    except:
        return 0.


def copydir(src, dst):
    if os.path.exists(dst):
        removedirs(dst)
    copytree(src, dst)


def remove(path):
    if os.path.exists(path):
        os.remove(path)


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def removedirs(path):
    if os.path.exists(path):
        rmtree(path)


def str_flags(flags):
    p = ''
    for key in np.sort(list(flags.keys())):
        p += str(key) + ':' + str(flags.get(key)._value) + '\n'
    return p


def rampup(step, rampup_length):
    p = tf.minimum(1.0, tf.cast(step, tf.float32) / rampup_length)
    return tf.nn.sigmoid(10.0*(p-0.5)) / sigmoid(5.0)

def save_images(images, size, path):
    if images.shape[3] == 1:
        images = np.concatenate([images, images, images], 3)
    images = np.clip(images, -1.0, 1.0)
    return scipy.misc.toimage(merge(images, size), cmin=-1, cmax=1).save(path)


def imread(path, is_grayscale=False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def imresize(image, resize=1):
    h, w = image.shape[0], image.shape[1]
    img = np.zeros((h * resize, w * resize, image.shape[2]))
    for i in range(h * resize):
        for j in range(w * resize):
            img[i, j] = image[i // resize, j // resize]
    return img


def merge(images, size, resize=3):
    h, w = images.shape[1] * resize, images.shape[2] * resize
    img = np.zeros((h * size[0], w * size[1], images.shape[3]))
    assert size[0] * size[1] == images.shape[0]
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = imresize(image, resize)
    return img


def center_crop(x, crop_h, crop_w=None, resize_w=64):
    h, w = x.shape[:2]
    if crop_w is None:
        crop_w = crop_h
    if crop_h == 0:
        crop_h = crop_w = min(h, w)
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return scipy.misc.imresize(x[j:j + crop_h, i:i + crop_w], [resize_w, resize_w])


def batch_resize(images, newHeight, newWidth):
    images_resized = np.zeros([images.shape[0], newHeight, newWidth, 3])
    for idx, image in enumerate(images):
        if (images.shape[3] == 1):
            image = np.concatenate([image, image, image], 2)
        images_resized[idx] = scipy.misc.imresize(image, [newHeight, newWidth], 'bilinear')
    return images_resized


def clip_truncated_normal(mean, stddev, shape, minval=None, maxval=None):
    if minval == None:
        minval = mean - 2 * stddev
    if maxval == None:
        maxval = mean + 2 * stddev
    return np.clip(np.random.normal(mean, stddev, shape), minval, maxval)


def collect(X, x, len):
    if isinstance(x, np.ndarray):
        if x.shape.__len__() == 1:
            x = x.reshape((1,) + x.shape)
        return x if X is None else np.concatenate([X, x], 0)[-len:]
    else:
        return [x] if X is None else (X + [x])[-len:]


def get_name(layer_name, cts):
    if not layer_name in cts:
        cts[layer_name] = 0
    name = layer_name + '_' + str(cts[layer_name])
    cts[layer_name] += 1
    return name


def shuffle_datas(datas):
    rand_indexes = np.random.permutation(datas.shape[0])
    shuffled_images = datas[rand_indexes]
    return shuffled_images


def shuffle_datas_and_labels(datas, labels):
    rand_indexes = np.random.permutation(datas.shape[0])
    shuffled_images = datas[rand_indexes]
    shuffled_labels = labels[rand_indexes]
    return shuffled_images, shuffled_labels


def data_gen_random(data, num_sample):
    while True:
        num_data = len(data)
        data_index = np.random.choice(num_data, num_sample, replace=True, p=num_data * [1 / num_data])
        yield data[data_index]


def data_gen_epoch(datas, batch_size, func=None, epoch=None):
    cur_epoch = 0

    while len(datas) < 100 * batch_size:
        datas = np.concatenate([datas, datas], axis=0)

    while True:
        np.random.shuffle(datas)
        for i in range(len(datas) // batch_size):
            if func is None:
                yield datas[i * batch_size:(i + 1) * batch_size]
            else:
                yield func(datas[i * batch_size:(i + 1) * batch_size])

        cur_epoch += 1
        if epoch is not None:
            if cur_epoch >= epoch:
                break


def labeled_data_gen_random(data, labels, num_sample):
    while True:
        num_data = len(data)
        index = np.random.choice(num_data, num_sample, replace=True, p=num_data * [1 / num_data])
        yield data[index], labels[index]


def labeled_data_gen_epoch(datas, labels, batch_size, func=None, epoch=None):
    cur_epoch = 0
    while True:
        rng_state = np.random.get_state()
        np.random.shuffle(datas)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)
        for i in range(len(datas) // batch_size):
            if func is None:
                yield (datas[i * batch_size:(i + 1) * batch_size], labels[i * batch_size:(i + 1) * batch_size])
            else:
                yield (func(datas[i * batch_size:(i + 1) * batch_size]), labels[i * batch_size:(i + 1) * batch_size])
        cur_epoch += 1
        if epoch is not None:
            if cur_epoch >= epoch:
                break


def random_augment_image_nchw(image, pad=4, data_format="NCHW"):

    if data_format=="NHWC":
        image = np.transpose(image, [2,0,1])

    init_shape = image.shape
    new_shape = [init_shape[0],
                 init_shape[1] + pad * 2,
                 init_shape[2] + pad * 2]
    zeros_padded = np.zeros(new_shape)
    zeros_padded[:, pad:init_shape[1] + pad, pad:init_shape[2] + pad] = image

    init_x = np.random.randint(0, pad * 2)
    init_y = np.random.randint(0, pad * 2)
    cropped = zeros_padded[:,
        init_x: init_x + init_shape[1],
        init_y: init_y + init_shape[2]]

    flip = random.getrandbits(1)
    if flip:
        cropped = cropped[:, :, ::-1]

    if data_format=="NHWC":
        cropped = np.transpose(cropped, [1,2,0])

    return cropped


def random_augment_image_nhwc(image, pad=4, data_format="NHWC"):

    if data_format=="NCHW":
        image = np.transpose(image, [1,2,0])

    init_shape = image.shape
    new_shape = [init_shape[0] + pad * 2,
                 init_shape[1] + pad * 2,
                 init_shape[2]]
    zeros_padded = np.zeros(new_shape)
    zeros_padded[pad:init_shape[0] + pad, pad:init_shape[1] + pad, :] = image

    init_x = np.random.randint(0, pad * 2)
    init_y = np.random.randint(0, pad * 2)
    cropped = zeros_padded[
        init_x: init_x + init_shape[0],
        init_y: init_y + init_shape[1],
        :]

    flip = random.getrandbits(1)
    if flip:
        cropped = cropped[:, ::-1, :]

    if data_format=="NCHW":
        cropped = np.transpose(cropped, [2,0,1])

    return cropped


def random_augment_all_images(initial_images, pad=4, data_format="NCHW"):
    new_images = np.zeros(initial_images.shape)
    for i in range(initial_images.shape[0]):
        new_images[i] = random_augment_image_nchw(initial_images[i], pad=pad, data_format=data_format)
    return new_images


def softmax(x):
    e_x = np.exp(x - np.max(x, 1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)