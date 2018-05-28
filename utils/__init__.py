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

def test_accuracy(logits, labels):
    logits_idx = tf.to_int32(tf.argmax(logits, axis=1))
    logits_idx = tf.reshape(logits_idx, shape=(cfg.batch_size,))
    correct_preds = tf.equal(tf.to_int32(labels), logits_idx)
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) / cfg.batch_size

    return accuracy

