"""
License: Apache-2.0
Author: Suofei Zhang | Hang Yu
E-mail: zhangsuofei at njupt.edu.cn | hangyu5 at illinois.edu
"""

import tensorflow as tf
from config import cfg
from models.cnn_baseline import build_cnn_baseline, cross_ent_loss
import time
import os
from utils import test_accuracy
import tensorflow.contrib.slim as slim
from utils import load_mscoco, test_accuracy, load_tless_split, load_fashion_mnist

# import logging
# import daiquiri

# daiquiri.setup(level=logging.DEBUG)
# logger = daiquiri.getLogger(__name__)


def main(args):
    """Get dataset hyperparameters."""
    # assert len(args) == 3 and isinstance(args[1], str) and isinstance(args[2], str)
    assert len(args) == 3 and isinstance(args[1], str) and args[2] in ["mscoco", "tless", 'fashion']
    experiment_name = args[1]
    dataset_name = args[2]

    """ GET DATA """
    if dataset_name == 'mscoco':
        test_dataset = load_mscoco('test', cfg, return_dataset=True)
        dataset_name = 'mscoco' 
        num_classes = 2
    elif dataset_name == 'tless':
        num_classes = 10
        _, _, test_dataset = load_tless_split(cfg, num_classes)

        # squash labels like so: turn original labels of [1, 55, 33, 33, 1, 33, 55] into [0, 2, 1, 1, 0, 1, 2]
        _, test_dataset.y = np.unique(np.asarray(val_dataset.y), return_inverse=True)
    elif dataset_name == "fashion":
        test_dataset = load_fashion_mnist(cfg, phase = 'test')
        num_classes = 10

    # dataset_name = args[1]
    # model_name = args[2]
    # dataset_size_train = dataset.X.shape[0]
    dataset_size_test = test_dataset.X.shape[0]
    D = test_dataset.X.shape[1]

    if cfg.greyscale:
        print('using greyscale')
        num_channels = 1 
    else:
        num_channels = 3 
    # create_inputs = get_create_inputs(
    #     dataset_name, is_train=False, epochs=cfg.epoch)

    """Set reproduciable random seed"""
    tf.set_random_seed(1234)

    with tf.Graph().as_default():
        # num_batches_per_epoch_train = int(dataset_size_train / cfg.batch_size)
        num_batches_test = int(dataset_size_test / cfg.batch_size * 0.1)

        batch_x = tf.placeholder(tf.float32, shape=(cfg.batch_size, D, D, num_channels), name="input")
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

            # coord = tf.train.Coordinator()
            # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            if not os.path.exists(cfg.test_logdir + '/cnn_baseline/{}/'.format(experiment_name)):
                os.makedirs(cfg.test_logdir + '/cnn_baseline/{}/'.format(experiment_name))
            summary_writer = tf.summary.FileWriter(
                cfg.test_logdir + '/cnn_baseline/{}/'.format(experiment_name), graph=sess.graph)  # graph=sess.graph, huge!

            files = os.listdir(cfg.logdir + '/cnn_baseline/{}/'.format(experiment_name))
            ckpt = tf.train.get_checkpoint_state(cfg.logdir + '/cnn_baseline/{}/best_checkpoint/'.format(experiment_name))
            for epoch in range(1, cfg.num_epochs):
                # requires a regex to adapt the loss value in the file name here
                #we should only have 1 
                # ckpt_re = ".ckpt-%d" % (num_batches_per_epoch_train * epoch)
                # for __file in files:
                #     if __file.endswith(".index"):
                #         ckpt = os.path.join(cfg.logdir + '/{}/{}/'.format(model_name, dataset_name), __file[:-6])
                #         print('ckpt is', ckpt)
                # ckpt = os.path.join(cfg.logdir, "model.ckpt-%d" % (num_batches_per_epoch_train * epoch))
                saver.restore(sess, ckpt.model_checkpoint_path)

                accuracy_sum = 0

                while test_dataset.has_next_batch():
                    test_batch = test_dataset.next_batch()
                    feed_dict = {batch_x: test_batch[0].astype(np.float32), batch_labels: test_batch[1]}
                    batch_acc_v, summary_str, step_out = sess.run([acc, summary_op, global_step], feed_dict=feed_dict)

                    print('%d batches are tested.' % step)
                    summary_writer.add_summary(summary_str, step)

                    accuracy_sum += batch_acc_v

                    step += 1

                ave_acc = accuracy_sum / num_batches_test
                print('the average accuracy in this epoch is %f' % ave_acc)

            # coord.join(threads)


if __name__ == "__main__":
    tf.app.run()
