"""
License: Apache-2.0
Author: Suofei Zhang | Hang Yu
E-mail: zhangsuofei at njupt.edu.cn | hangyu5 at illinois.edu
"""

import tensorflow as tf
from config import cfg, get_dataset_size_train, get_dataset_size_test, get_num_classes, get_create_inputs
from models.cnn_baseline import build_cnn_baseline, cross_ent_loss

import time
import os
from utils import test_accuracy
import tensorflow.contrib.slim as slim

import logging
import daiquiri

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)


def main(args):
    """Get dataset hyperparameters."""
    # assert len(args) == 3 and isinstance(args[1], str) and isinstance(args[2], str)
    dataset = load_mscoco(cfg.phase, cfg, return_dataset=True)
    dataset_name = 'mscoco'
    experiment_name = args[1]

    # dataset_name = args[1]
    # model_name = args[2]
    # dataset_size_train = dataset.X.shape[0]
    dataset_size_test = dataset.X.shape[0]
    num_classes = cfg.num_classes
    # create_inputs = get_create_inputs(
    #     dataset_name, is_train=False, epochs=cfg.epoch)

    """Set reproduciable random seed"""
    tf.set_random_seed(1234)

    with tf.Graph().as_default():
        # num_batches_per_epoch_train = int(dataset_size_train / cfg.batch_size)
        num_batches_test = int(dataset_size_test / cfg.batch_size * 0.1)

        batch_x = tf.placeholder(tf.float32, shape=(cfg.batch_size, D, D, 3), name="input")
        batch_labels = tf.placeholder(tf.int32, shape=(cfg.batch_size), name="labels")        
        batch_x = slim.batch_norm(batch_x, center=False, is_training=False, trainable=False)
        batch_x_squash = tf.divide(batch_x, 255.)
        batch_x_norm = slim.batch_norm(batch_x, center=False, is_training=True, trainable=True)
        output = build_cnn_baseline(batch_x_norm, is_train=True, num_classes=num_classes)
        acc = test_accuracy(output, batch_labels)
    

        batch_acc = test_accuracy(output, batch_labels)
        saver = tf.train.Saver()

        step = 0

        summaries = []
        summaries.append(tf.summary.scalar('accuracy', batch_acc))
        summary_op = tf.summary.merge(summaries)

        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=False)) as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            if not os.path.exists(cfg.test_logdir + '/{}/{}/'.format(model_name, dataset_name)):
                os.makedirs(cfg.test_logdir + '/{}/{}/'.format(model_name, dataset_name))
            summary_writer = tf.summary.FileWriter(
                cfg.test_logdir + '/{}/{}/'.format(model_name, dataset_name), graph=sess.graph)  # graph=sess.graph, huge!

            files = os.listdir(cfg.logdir + '/{}/{}/'.format(model_name, dataset_name))

            for epoch in range(1, cfg.epoch):
                # requires a regex to adapt the loss value in the file name here
                #we should only have 1 
                # ckpt_re = ".ckpt-%d" % (num_batches_per_epoch_train * epoch)
                # for __file in files:
                #     if __file.endswith(".index"):
                #         ckpt = os.path.join(cfg.logdir + '/{}/{}/'.format(model_name, dataset_name), __file[:-6])
                #         print('ckpt is', ckpt)
                ckpt = tf.train.get_checkpoint_state(cfg.logdir + '/cnn_baseline/{}/'.format(dataset_name))
                # ckpt = os.path.join(cfg.logdir, "model.ckpt-%d" % (num_batches_per_epoch_train * epoch))
                saver.restore(sess, ckpt.model_checkpoint_path)

                accuracy_sum = 0
                for i in range(num_batches_test):
                    batch_acc_v, summary_str = sess.run([batch_acc, summary_op])
                    print('%d batches are tested.' % step)
                    summary_writer.add_summary(summary_str, step)

                    accuracy_sum += batch_acc_v

                    step += 1

                ave_acc = accuracy_sum / num_batches_test
                print('the average accuracy is %f' % ave_acc)

            coord.join(threads)


if __name__ == "__main__":
    tf.app.run()
