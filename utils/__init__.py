import os
import scipy
import numpy as np
import tensorflow as tf

from dr_config import cfg
from .dataset import Dataset, TLessDataset
from .dataset import load_mscoco, load_tless_split



def get_batch_data(batch_size, num_threads):
    train, val, _ = load_tless_split(cfg, num_classes=2)
    trX, trY = train.X, train.y
    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=num_threads,
                                  batch_size=batch_size,
                                  capacity=batch_size * 64,
                                  min_after_dequeue=batch_size * 32,
                                  allow_smaller_final_batch=False)

    return(X, Y)


def save_images(imgs, size, path):
    '''
    Args:
        imgs: [batch_size, image_height, image_width]
        size: a list with tow int elements, [image_height, image_width]
        path: the path to save images
    '''
    imgs = (imgs + 1.) / 2  # inverse_transform
    return(scipy.misc.imsave(path, mergeImgs(imgs, size)))


def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs


# For version compatibility
def reduce_sum(input_tensor, axis=None, keepdims=False):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims)
    except:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims)


# For version compatibility
def softmax(logits, axis=None):
    try:
        return tf.nn.softmax(logits, axis=axis)
    except:
        return tf.nn.softmax(logits, dim=axis)


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
