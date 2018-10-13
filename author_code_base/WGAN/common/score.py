import sys
import tarfile
import math
from os import path

from common.utils import *
import scipy as sp

SOURCE_DIR = path.dirname(path.dirname(path.abspath(__file__))) + '/'


class InceptionScore:

    @staticmethod
    def inception_score_KL(preds):
        preds = preds + 1e-18
        inps_avg_preds = np.mean(preds, 0, keepdims=True)
        inps_KLs = np.sum(preds * (np.log(preds) - np.log(inps_avg_preds)), 1)
        inception_score = np.exp(np.mean(inps_KLs))
        return inception_score

    @staticmethod
    def inception_score_H(preds): # inception_score_KL == inception_score_H
        preds = preds + 1e-18
        inps_avg_preds = np.mean(preds, 0)
        H_per = np.mean(-np.sum(preds * np.log(preds), 1))
        H_avg = -np.sum(inps_avg_preds * np.log(inps_avg_preds), 0)
        return np.exp(H_avg - H_per), H_per, H_avg

    @staticmethod
    def inception_score_split_std(icp_preds, split_n=10):
        icp_preds = icp_preds + 1e-18
        scores = []
        for i in range(split_n):
            part = icp_preds[(i * icp_preds.shape[0] // split_n):((i + 1) * icp_preds.shape[0] // split_n), :]
            scores.append(InceptionScore.inception_score_KL(part))
        return np.mean(scores), np.std(scores), scores


class AMScore:

    @staticmethod
    def am_per_score(preds):
        preds = preds + 1e-18
        am_per = -np.sum(preds * np.log(preds), 1)
        return am_per

    @staticmethod
    def am_score(preds, ref_preds):

        preds = preds + 1e-18
        avg_preds = np.mean(preds, 0)
        ref_avg_preds = np.mean(ref_preds, 0)
        am_per = np.mean(-np.sum(preds * np.log(preds), 1))
        am_avg = -np.sum(ref_avg_preds * np.log(avg_preds / ref_avg_preds), 0)
        am_score = am_per + am_avg

        return am_score, am_per, am_avg

    @staticmethod
    def am_score_split_std(preds, ref_preds, split_n=10):
        scores = []
        for i in range(split_n):
            part = preds[(i * preds.shape[0] // split_n):((i + 1) * preds.shape[0] // split_n), :]
            scores.append(AMScore.am_score(part, ref_preds)[0])
        return np.mean(scores), np.std(scores), scores


class FID:

    @staticmethod
    def get_stat(activations):
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

    @staticmethod
    def get_FID_with_stat(mu, sigma, ref_mu, ref_sigma, useTrace=True): #useTrace=True --> FD; useTrace=False simply a two moment matching measurement.
        if useTrace:
            m = np.square(mu - ref_mu).sum()
            s = sp.linalg.sqrtm(np.dot(sigma, ref_sigma))
            s = np.trace(sigma + ref_sigma - 2 * s)
            dist = m + s
            if np.isnan(dist):
                print('nan fid')
                return m + 100
        else:
            m = np.square(mu - ref_mu).sum()
            s = np.square(sigma - ref_sigma).sum()
            dist = m + s
        return dist

    @staticmethod
    def get_FID_with_activations(activations, ref_activation):
        mu, sigma = FID.get_stat(activations)
        ref_mu, ref_sigma = FID.get_stat(ref_activation)
        return FID.get_FID_with_stat(mu, sigma, ref_mu, ref_sigma)


class PreTrainedInception:

    def __init__(self):

        self.batch_size = 100 # It does not affect the accuracy. Small batch size need less memory while bit slower

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True

        self.inception_graph = tf.Graph()
        self.inception_sess = tf.Session(config=config, graph=self.inception_graph)

        self._init_model_()

    def get_preds(self, inps):

        inps_s = np.array(inps) * 128.0 + 128

        icp_preds_w = []
        icp_preds_b = []
        activations = []
        f_batches = int(math.ceil(float(inps_s.shape[0]) / float(self.batch_size)))
        for i in range(f_batches):
            inp = inps_s[(i * self.batch_size): min((i + 1) * self.batch_size, inps_s.shape[0])]
            pred_w, pred_b, activation = self.inception_sess.run([self.inception_softmax_w, self.inception_softmax_b, self.activation], {'ExpandDims:0': inp})
            icp_preds_w.append(pred_w)
            icp_preds_b.append(pred_b)
            activations.append(activation)

        icp_preds_w = np.concatenate(icp_preds_w, 0)
        icp_preds_b = np.concatenate(icp_preds_b, 0)
        activations = np.concatenate(activations, 0)
        activations = activations.reshape([activations.shape[0], -1])

        return icp_preds_w, activations

    def _init_model_(self):

        MODEL_DIR = SOURCE_DIR + 'pretrained_model/inception/'
        DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

        makedirs(MODEL_DIR)
        filename = DATA_URL.split('/')[-1]
        filepath = os.path.join(MODEL_DIR, filename)

        if not os.path.exists(os.path.join(MODEL_DIR, 'classify_image_graph_def.pb')):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))

            if sys.version_info[0] >= 3:
                from urllib.request import urlretrieve
            else:
                from urllib import urlretrieve

            filepath, _ = urlretrieve(DATA_URL, filepath, _progress)
            print('\nSuccesfully downloaded', filename, os.stat(filepath).st_size, 'bytes.')
            tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)

        with tf.gfile.FastGFile(os.path.join(MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with self.inception_graph.as_default():
            tf.import_graph_def(graph_def, name='')

        # Works with an arbitrary minibatch size.
        ops = self.inception_graph.get_operations()
        for op_idx, op in enumerate(ops):
            for o in op.outputs:
                shape = o.get_shape()
                shape = [s.value for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o._shape = tf.TensorShape(new_shape) # works in tensorflow-1.5; it does not change NodeDef and hence GraphDef.
                # o._shape_val = tf.TensorShape(new_shape) # works in tensorflow-1.5; it does not change NodeDef and hence GraphDef.
                # o.set_shape(tf.TensorShape(new_shape)) # failed in tensorflow-1.9. it does not change the shape. because set_shape() will not change the shape from 'known' to 'unknown'. tensorflow-1.9

        pool3 = self.inception_graph.get_tensor_by_name("pool_3:0")
        w = self.inception_graph.get_tensor_by_name("softmax/weights:0")
        output = tf.matmul(tf.reshape(pool3, [-1, 2048]), w)
        self.inception_softmax_w = tf.nn.softmax(output)

        b = self.inception_graph.get_tensor_by_name("softmax/biases:0")
        output = tf.add(output, b)
        self.inception_softmax_b = tf.nn.softmax(output)

        self.activation = pool3


class PreTrainedDenseNet:

    def __init__(self, num_class=10, batch_size=128):

        self.num_class = num_class
        self.batch_size = batch_size

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True

        self.dense_graph = tf.Graph()
        self.dense_sess = tf.Session(config=config, graph=self.dense_graph)

        self._init_model_()

    def get_preds(self, images, ref_images=None):

        if ref_images is None:
            ref_images = images

        means = []
        stds = []
        for ch in range(ref_images.shape[-1]):
            means.append(np.mean(ref_images[:, :, :, ch]))
            stds.append(np.std(ref_images[:, :, :, ch]))

        images_standardized = np.zeros_like(images)
        for i in range(images.shape[-1]):
            images_standardized[:, :, :, i] = ((images[:, :, :, i] - means[i]) / stds[i])

        preds = []
        activations = []
        f_batches = int(math.ceil(float(images_standardized.shape[0]) / float(self.batch_size)))
        for i in range(f_batches):
            image = images_standardized[(i * self.batch_size): min((i + 1) * self.batch_size, images_standardized.shape[0])]
            pred, activation = self.dense_sess.run([self.preds, self.activations], {self.inputs: image, self.is_training: False})
            preds.append(pred)
            activations.append(activation)

        preds = np.concatenate(preds, 0)
        activations = np.concatenate(activations, 0)

        return preds, activations

    def _init_model_(self):
        with self.dense_graph.as_default():
            saver = tf.train.import_meta_graph(SOURCE_DIR + 'pretrained_model/densenet/DenseNet-BC_growth_rate=40_depth=40_dataset_C10+/model.chkpt.meta')
            saver.restore(self.dense_sess, SOURCE_DIR + 'pretrained_model/densenet/DenseNet-BC_growth_rate=40_depth=40_dataset_C10+/model.chkpt')
        self.preds = self.dense_graph.get_tensor_by_name('Softmax:0')
        self.activations = self.dense_graph.get_tensor_by_name('Transition_to_classes/Reshape:0')
        self.inputs = self.dense_graph.get_tensor_by_name('input_images:0')
        self.is_training = self.dense_graph.get_tensor_by_name('Placeholder:0')


class MSSSIM:  # tf version of MSSIM

    def _tf_fspecial_gauss(self, size, sigma):
        """Function to mimic the 'fspecial' gaussian MATLAB function
        """

        radius = size // 2
        offset = 0.0
        start, stop = -radius, radius + 1
        if size % 2 == 0:
            offset = 0.5
            stop -= 1

        x_data, y_data = np.mgrid[offset + start:stop, offset + start:stop]  # -size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        assert len(x_data) == size

        x_data = np.expand_dims(x_data, axis=-1)
        x_data = np.expand_dims(x_data, axis=-1)

        y_data = np.expand_dims(y_data, axis=-1)
        y_data = np.expand_dims(y_data, axis=-1)

        x = tf.constant(x_data, dtype=tf.float32)
        y = tf.constant(y_data, dtype=tf.float32)

        g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
        return g / tf.reduce_sum(g)

    def tf_ssim(self, img1, img2, L=2, mean_metric=True, filter_size=11, sigma=1.5):

        _, height, width, _ = img1.shape.as_list()

        size = min(filter_size, height, width)
        sigma = size / 11.0 * sigma

        window = self._tf_fspecial_gauss(size, sigma)  # window shape [size, size]

        K1 = 0.01
        K2 = 0.03
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2

        def filter(img):
            return tf.concat([tf.nn.conv2d(img[:,:,:,0:1], window, strides=[1, 1, 1, 1], padding='VALID'),tf.nn.conv2d(img[:,:,:,1:2], window, strides=[1, 1, 1, 1], padding='VALID'),tf.nn.conv2d(img[:,:,:,2:3], window, strides=[1, 1, 1, 1], padding='VALID')], 3)

        mu1 = filter(img1)
        mu2 = filter(img2)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = filter(img1 * img1)- mu1_sq
        sigma2_sq = filter(img2 * img2) - mu2_sq
        sigma12 = filter(img1 * img2) - mu1_mu2
        sigma12 = tf.abs(sigma12)

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = v1 / v2
        ssim = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1) * cs

        if mean_metric:
            ssim = tf.reduce_mean(ssim)
            cs = tf.reduce_mean(cs)

        return ssim, cs

    def tf_ms_ssim(self, img1, img2, mean_metric=True, level=5):

        #assert level >= 1 and level <= 5
        weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
        #weight = weight[-level:] / tf.reduce_sum(weight[-level:])
        #weight = tf.ones([level], dtype=tf.float32) / level
        #weight = tf.Print(weight, [weight])

        mssim = []
        mcs = []
        for l in range(level):
            ssim, cs = self.tf_ssim(img1, img2, mean_metric=True)
            mssim.append(ssim)
            mcs.append(cs)
            img1 = tf.nn.avg_pool(img1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            img2 = tf.nn.avg_pool(img2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        # list to tensor of dim D+1
        mssim = tf.stack(mssim, axis=0)
        mcs = tf.stack(mcs, axis=0)

        value = (tf.reduce_prod(mcs[0:level - 1] ** weight[0:level - 1]) * (mssim[level - 1] ** weight[level - 1]))

        if mean_metric:
            value = tf.reduce_mean(value)

        return value

    def __init__(self):

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True

        self.ms_ssim_graph = tf.Graph()
        self.ms_ssim_sess = tf.Session(config=config, graph=self.ms_ssim_graph)

        with self.ms_ssim_graph.as_default():
            self.image1 = tf.placeholder(tf.float32, shape=[1, 32, 32, 3])
            self.image2 = tf.placeholder(tf.float32, shape=[1, 32, 32, 3])
            self.msssim_value = self.tf_ms_ssim(self.image1, self.image2)

    def msssim(self, class_images, num_classes, count):

        class_images += 1

        classes_score = []
        for i in range(num_classes):
            scores = []
            for x1 in range(count//2):
                x2 = count - x1 - 1
                score = self.ms_ssim_sess.run(self.msssim_value, feed_dict={self.image1: class_images[count * i + x1:count * i + x1 + 1], self.image2: class_images[count * i + x2:count * i + x2 + 1]})
                #score = MultiScaleSSIM(class_images[count * i + x1:count * i + x1 + 1], class_images[count * i + x2:count * i + x2 + 1], 2)
                scores.append(score)
            classes_score.append(scores)

        return [mean(scores) for scores in classes_score], [std(scores) for scores in classes_score]