import os
import scipy
import numpy as np
import tensorflow as tf

from dataset import Dataset
from keras.datasets import cifar10, cifar100

import logging

def create_inputs_cifar10(is_train, config):
    tr_x, tr_y = load_cifar10(is_train)
    data_queue = tf.train.slice_input_producer([tr_x, tr_y], capacity=64 * 8)
    x, y = tf.train.shuffle_batch(data_queue, num_threads=8, batch_size=config.batch_size, capacity=config.batch_size * 64,
                                  min_after_dequeue=config.batch_size * 32, allow_smaller_final_batch=False)
    return (x, y)

def load_cifar10(is_training):
    from keras import backend
    assert(backend.image_data_format() == 'channels_last')
    if is_training:
        return cifar10.load_data()[0]
    else:
        return cifar10.load_data()[1]

def create_inputs_cifar10(is_training, config):
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