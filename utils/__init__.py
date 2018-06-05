import os
import scipy
import numpy as np
import tensorflow as tf

from norb_config import cfg
from .dataset import Dataset, TLessDataset
from .dataset import load_mscoco, load_tless_split, load_norb


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
