import os
import scipy
import numpy as np
import tensorflow as tf

from config import cfg
from keras.datasets import cifar10, cifar100
from keras import backend as K


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

def get_create_inputs(dataset_name: str, is_train: bool, epochs: int, config):
    options = {'cifar10': lambda: create_inputs_cifar10(is_train, config),
               'cifar100': lambda: create_inputs_cifar100(is_train, config),
               'mscoco': lambda: create_inputs_mscoco(is_train, config)}
    return options[dataset_name]
