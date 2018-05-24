import os
import scipy
import numpy as np
import tensorflow as tf

from config import cfg
from keras.datasets import cifar10, cifar100
from keras import backend as K


def create_inputs_mnist(is_train):
    tr_x, tr_y = load_mnist(cfg.dataset, is_train)
    data_queue = tf.train.slice_input_producer([tr_x, tr_y], capacity=64 * 8)
    x, y = tf.train.shuffle_batch(data_queue, num_threads=8, batch_size=cfg.batch_size, capacity=cfg.batch_size * 64,
                                  min_after_dequeue=cfg.batch_size * 32, allow_smaller_final_batch=False)

    return (x, y)


def create_inputs_fashion_mnist(is_train):
    tr_x, tr_y = load_mnist(cfg.dataset_fashion_mnist, is_train)
    data_queue = tf.train.slice_input_producer([tr_x, tr_y], capacity=64 * 8)
    x, y = tf.train.shuffle_batch(data_queue, num_threads=8, batch_size=cfg.batch_size, capacity=cfg.batch_size * 64,
                                  min_after_dequeue=cfg.batch_size * 32, allow_smaller_final_batch=False)

    return (x, y)


def load_mnist(path, is_training):
    fd = open(os.path.join(cfg.dataset, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

    fd = open(os.path.join(cfg.dataset, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.int32)

    fd = open(os.path.join(cfg.dataset, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float32)

    fd = open(os.path.join(cfg.dataset, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.int32)

    # normalization and convert to a tensor [60000, 28, 28, 1]
    # trX = tf.convert_to_tensor(trX, tf.float32)
    # teX = tf.convert_to_tensor(teX, tf.float32)

    # => [num_samples, 10]
    # trY = tf.one_hot(trY, depth=10, axis=1, dtype=tf.float32)
    # teY = tf.one_hot(teY, depth=10, axis=1, dtype=tf.float32)

    if is_training:
        return trX, trY
    else:
        return teX, teY


def create_inputs_cifar10(is_train):
    tr_x, tr_y = load_cifar10(is_train)
    data_queue = tf.train.slice_input_producer([tr_x, tr_y], capacity=64 * 8)
    x, y = tf.train.shuffle_batch(data_queue, num_threads=8, batch_size=cfg.batch_size, capacity=cfg.batch_size * 64,
                                  min_after_dequeue=cfg.batch_size * 32, allow_smaller_final_batch=False)

    return (x, y)


def load_cifar10(is_training):
    # https://keras.io/datasets/
    assert(K.image_data_format() == 'channels_last')
    if is_training:
        return cifar10.load_data()[0]
    else:
        return cifar10.load_data()[1]

def create_inputs_mscoco(is_training, config):
    tr_x, tr_y = load_mscoco(is_training, config)
    data_queue = tf.train.slice_input_producer([tr_x, tr_y], capacity=64 * 8)
    x, y = tf.train.shuffle_batch(data_queue, num_threads=8, batch_size=config.batch_size, capacity=config.batch_size * 64,
                                  min_after_dequeue=config.batch_size * 32, allow_smaller_final_batch=False)
    return (x, y)

def load_mscoco(is_training, config):
    if is_training:
        if config.use_masked:
            data = Dataset("data/train/images/masked")
        else:
            data = Dataset("data/train/images/train2014")
    else:
        if config.use_masked:
            data = Dataset("data/test/images/masked")
        else:
            data = Dataset("data/test/images/train2014")

    return (data.X, data.y)

def create_inputs_cifar100(is_train):
    tr_x, tr_y = load_cifar100(is_train)
    data_queue = tf.train.slice_input_producer([tr_x, tr_y], capacity=64 * 8)
    x, y = tf.train.shuffle_batch(data_queue, num_threads=8, batch_size=cfg.batch_size, capacity=cfg.batch_size * 64,
                                  min_after_dequeue=cfg.batch_size * 32, allow_smaller_final_batch=False)

    return (x, y)


def load_cifar100(is_training):
    # https://keras.io/datasets/
    # https://www.cs.toronto.edu/~kriz/cifar.html:
    # "Each image comes with a 'fine' label (the class to which it belongs)
    # and a 'coarse' label (the superclass to which it belongs)."
    assert(K.image_data_format() == 'channels_last')
    if is_training:
        return cifar100.load_data(label_mode='fine')[0]
    else:
        return cifar100.load_data(label_mode='fine')[1]

def get_create_inputs(dataset_name: str, is_train: bool, epochs: int):
    options = {'mnist': lambda: create_inputs_mnist(is_train),
               'fashion_mnist': lambda: create_inputs_mnist(is_train),
               'cifar10': lambda: create_inputs_cifar10(is_train),
               'cifar100': lambda: create_inputs_cifa100(is_train)}
    return options[dataset_name]
