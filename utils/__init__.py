import os
import scipy
import numpy as np
import tensorflow as tf

from config import cfg
from keras.datasets import cifar10, cifar100
from keras import backend as K
from .dataset import Dataset

def load_mscoco(dataset_type, config, num=None, return_dataset=False):
    if dataset_type == 'train':
        if config.use_masked:
            data = Dataset("data/train/images/masked", is_train=True, batch_size=config.batch_size, num=num)
        else:
            data = Dataset("data/train/images/train2014", batch_size=config.batch_size, num=num, is_train=True)
    elif dataset_type == 'test':
        if config.use_masked:
            data = Dataset("data/test/images/masked", batch_size=config.batch_size, num=num)
        else:
            data = Dataset("data/test/images/test2014", batch_size=config.batch_size, num=num)
    elif dataset_type == 'val':
        if config.use_masked:
            data = Dataset("data/val/images/masked", batch_size=config.batch_size, num=num)
        else:
            data = Dataset("data/val/images/val2014", batch_size=config.batch_size, num=num)
    else:
        raise ValueError("Dataset type must be one of 'train', 'test', or 'val'")

    if return_dataset:
        return data

    return (data.X, data.y)

 
def create_inputs_mscoco(is_training, config):
    tr_x, tr_y = load_mscoco(is_training, config)
    data_queue = tf.train.slice_input_producer([tr_x, tr_y], capacity=64 * 8)
    x, y = tf.train.shuffle_batch(data_queue, num_threads=8, batch_size=config.batch_size, capacity=config.batch_size * 64,
                                  min_after_dequeue=config.batch_size * 32, allow_smaller_final_batch=False)
    return (x, y)

def test_accuracy(logits, labels):
    logits_idx = tf.to_int32(tf.argmax(logits, axis=1))
    logits_idx = tf.reshape(logits_idx, shape=(cfg.batch_size,))
    correct_preds = tf.equal(tf.to_int32(labels), logits_idx)
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) / cfg.batch_size

    return accuracy

