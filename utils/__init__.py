import os
import scipy
import numpy as np
import tensorflow as tf

from config import cfg
from . import mnist
from .dataset import Dataset

def load_mscoco(dataset_type, config, num=None, return_dataset=False):
    if dataset_type == 'train':
        data = Dataset("data/train/images/simple-2", is_train=True, batch_size=config.batch_size, num=num, greyscale=config.greyscale)
    elif dataset_type == 'test':
        data = Dataset("data/test/images/simple-2", batch_size=config.batch_size, num=num, greyscale=config.greyscale)
    elif dataset_type == 'val':
        data = Dataset("data/val/images/simple-2", is_train=True, batch_size=config.batch_size, num=num, greyscale=config.greyscale)
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

def one_hot_encode(labels, num_classes):
    num_labels = len(labels)
    one_hot_labels = np.zeros((num_labels, num_classes))
    one_hot_labels[np.arange(num_labels), labels] = 1 # one hot encode that shit 
    return one_hot_labels
