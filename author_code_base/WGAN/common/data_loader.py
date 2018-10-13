from os import path
import pickle, sys
from common.utils import *

SOURCE_DIR = path.dirname(path.dirname(path.abspath(__file__))) + '/'


def load_cifar10():

    def download_cifar10(data_dir):

        import os, sys,  tarfile

        if sys.version_info[0] >= 3:
            from urllib.request import urlretrieve
        else:
            from urllib import urlretrieve

        DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

        makedirs(data_dir)

        filename = DATA_URL.split('/')[-1]
        filepath = os.path.join(data_dir, filename)

        remove(filepath)
        removedirs(data_dir + '/cifar-10-batches-py/')

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))

        filepath, _ = urlretrieve(DATA_URL, filepath, _progress)
        print('\nSuccesfully downloaded', filename, os.stat(filepath).st_size, 'bytes.')

        tarfile.open(filepath, 'r:gz').extractall(data_dir)

    def unpickle(file):
        fo = open(file, 'rb')
        if sys.version_info[0] >= 3:
            dict = pickle.load(fo, encoding='bytes')
        else:
            dict = pickle.load(fo)
        fo.close()
        return dict

    data_dir = SOURCE_DIR + 'dataset/cifar-10-batches-py/'
    if not os.path.exists(data_dir):
        download_cifar10(SOURCE_DIR + 'dataset/')

    try:
        trfilenames = [os.path.join(data_dir, 'data_batch_%d' % i) for i in range(1, 6)]
        tefilenames = [os.path.join(data_dir, 'test_batch')]

        data_X = []
        data_Y = []

        test_X = []
        test_Y = []

        for files in trfilenames:
            dict = unpickle(files)
            data_X.append(dict.get(b'data'))
            data_Y.append(dict.get(b'labels'))

        for files in tefilenames:
            dict = unpickle(files)
            test_X.append(dict.get(b'data'))
            test_Y.append(dict.get(b'labels'))

        data_X = np.concatenate(data_X, 0)
        data_X = np.reshape(data_X, [-1, 3, 32, 32])
        #data_X = np.transpose(data_X, [0, 2, 3, 1])
        data_X = (data_X - 127.5) / 128.0
        data_Y = np.concatenate(data_Y, 0)
        data_Y = np.reshape(data_Y, [len(data_Y)]).astype(np.int32)

        test_X = np.concatenate(test_X, 0)
        test_X = np.reshape(test_X, [-1, 3, 32, 32])
        #test_X = np.transpose(test_X, [0, 2, 3, 1])
        test_X = (test_X - 127.5) / 128.0
        test_Y = np.concatenate(test_Y, 0)
        test_Y = np.reshape(test_Y, [len(test_Y)]).astype(np.int32)

        # data_X = np.resize(data_X[:100], [100, 3, 1024, 1024])
        # test_X = np.resize(test_X[:100], [100, 3, 1024, 1024])

        return data_X, data_Y, test_X, test_Y

    except Exception as e:
        print('Failed: ' + str(e))
        download_cifar10(data_dir)
        return load_cifar10()


def load_mnist(useX32=True, useC3=False):

    def download_mnist(data_dir):

        import subprocess
        removedirs(data_dir)
        makedirs(data_dir)

        url_base = 'http://yann.lecun.com/exdb/mnist/'
        file_names = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
        for file_name in file_names:
            url = (url_base + file_name).format(**locals())
            print(url)
            out_path = os.path.join(data_dir, file_name)
            cmd = ['curl', url, '-o', out_path]
            subprocess.call(cmd)
            cmd = ['gzip', '-d', out_path]
            subprocess.call(cmd)

    data_dir = SOURCE_DIR + 'dataset/mnist/'
    if not os.path.exists(data_dir):
        download_mnist(data_dir)

    try:
        fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trX = loaded[16:].reshape((60000, 1, 28, 28)).astype(np.float)

        fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 1, 28, 28)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)

        trY = np.asarray(trY)
        teY = np.asarray(teY)

        if useX32:
            trX32 = np.zeros([len(trX), 1, 32, 32])
            trX32[:, :, 2:30, 2:30] = trX
            trX = trX32

            teX32 = np.zeros([len(teX), 1, 32, 32])
            teX32[:, :, 2:30, 2:30] = teX
            teX = teX32

        if useC3:
            trX = np.concatenate([trX, trX, trX], 1)
            teX = np.concatenate([teX, teX, teX], 1)

        trY = (np.reshape(trY, [len(trY)])).astype(np.int32)
        teY = (np.reshape(teY, [len(teY)])).astype(np.int32)

        return (trX - 127.5) / 128.0, trY, (teX - 127.5) / 128.0, teY

    except Exception as e:
        print('Failed: ' + str(e))
        download_mnist(data_dir)
        return load_mnist(useX32)


def load_toy_data(n_mixture=8, std=0.01, radius=1.0, pts_per_mixture=1000000, add_middle=False):

    thetas = np.linspace(0, 2 * np.pi, n_mixture + 1)[:-1]
    xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
    cov = std * np.eye(2)
    stds = []
    weights = []

    X = np.zeros(((n_mixture + 0) * pts_per_mixture, 2))
    Y = np.zeros(((n_mixture + 0) * pts_per_mixture))

    for i in range(n_mixture):
        mean = np.array([xs[i], ys[i]])
        pts = np.random.multivariate_normal(mean, cov, pts_per_mixture)
        X[i * pts_per_mixture: (i + 1) * pts_per_mixture, :] = pts
        Y[i * pts_per_mixture: (i + 1) * pts_per_mixture] = i
        stds.append(std)
        weights.append(1.0/n_mixture)

    if add_middle:
        mean = np.array([0, 0])
        pts = np.random.multivariate_normal(mean, cov, pts_per_mixture)
        X[n_mixture * pts_per_mixture: (n_mixture + 1) * pts_per_mixture, :] = pts
        Y[n_mixture * pts_per_mixture: (n_mixture + 1) * pts_per_mixture] = n_mixture

    return X, Y.astype(np.int32), np.copy(X), np.copy(Y).astype(np.int32)


def load_toy_data_cov(n_mixture=8, std=0.01, radius=1.0, pts_per_mixture=1000000, add_middle=False):

    thetas = np.linspace(0, 2 * np.pi, n_mixture + 1)[:-1]
    xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
    cov = std * np.eye(2)
    stds = []
    weights = []

    X = np.zeros(((n_mixture + 0) * pts_per_mixture, 2))
    Y = np.zeros(((n_mixture + 0) * pts_per_mixture))

    for i in range(n_mixture):
        mean = np.array([xs[i], ys[i]])
        pts = np.random.multivariate_normal(mean, cov if i%2==0 else cov / 4, pts_per_mixture)
        X[i * pts_per_mixture: (i + 1) * pts_per_mixture, :] = pts
        Y[i * pts_per_mixture: (i + 1) * pts_per_mixture] = i
        stds.append(std if i % 2 == 0 else std / 4)
        weights.append(1.0/n_mixture)

    if add_middle:
        mean = np.array([0, 0])
        pts = np.random.multivariate_normal(mean, cov, pts_per_mixture)
        X[n_mixture * pts_per_mixture: (n_mixture + 1) * pts_per_mixture, :] = pts
        Y[n_mixture * pts_per_mixture: (n_mixture + 1) * pts_per_mixture] = n_mixture

    return X, Y.astype(np.int32), np.copy(X), np.copy(Y).astype(np.int32)


def load_toy_data_weight(n_mixture=8, std=0.01, radius=1.0, pts_per_mixture=1000000, add_middle=False):

    thetas = np.linspace(0, 2 * np.pi, n_mixture + 1)[:-1]
    xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
    cov = std * np.eye(2)
    stds = []
    weights = []

    X = np.zeros((5 * pts_per_mixture, 2))
    Y = np.zeros((5 * pts_per_mixture))

    tolsamples = 0
    for i in range(n_mixture):
        mean = np.array([xs[i], ys[i]])
        cursamples = pts_per_mixture if i % 2 == 0 else pts_per_mixture // 4
        pts = np.random.multivariate_normal(mean, cov, cursamples)
        X[tolsamples:tolsamples + cursamples] = pts
        Y[tolsamples:tolsamples + cursamples] = i
        stds.append(std)
        weights.append(1.0 / 5 if i % 2 == 0 else 1.0 / 20)
        tolsamples += cursamples

    if add_middle:
        mean = np.array([0, 0])
        pts = np.random.multivariate_normal(mean, cov, pts_per_mixture)
        X[n_mixture * pts_per_mixture: (n_mixture + 1) * pts_per_mixture, :] = pts
        Y[n_mixture * pts_per_mixture: (n_mixture + 1) * pts_per_mixture] = n_mixture

    return X, Y.astype(np.int32), np.copy(X), np.copy(Y).astype(np.int32)