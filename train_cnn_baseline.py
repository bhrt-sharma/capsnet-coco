import tensorflow as tf
import tensorflow.contrib.slim as slim
from config import cfg
from utils import create_inputs_mscoco, load_mscoco
import time
import numpy as np
import sys
import os
from tqdm import tqdm
from models.cnn_baseline import build_cnn_baseline, cross_ent_loss


def main(args):
    tf.set_random_seed(1234)

    """ GET DATA """
    dataset = load_mscoco('train', cfg, return_dataset=True)
    N, D = dataset.X.shape[0], dataset.X.shape[1]
    num_classes = int(max(dataset.y))
    num_batches_per_epoch = int(N / cfg.batch_size)

    """ SET UP INITIAL VARIABLES"""
    initial_learning_rate = 1e-3 # maybe move this to config?
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    lrn_rate = tf.maximum(tf.train.exponential_decay(initial_learning_rate, global_step, num_batches_per_epoch, 0.8), 1e-5)
    tf.summary.scalar('learning_rate', lrn_rate)
    opt = tf.train.AdamOptimizer(lrn_rate)

    """ DEFINE DATA FLOW """
    batch_x = tf.placeholder(tf.float32, shape=(cfg.batch_size, D, D, 3))
    batch_labels = tf.placeholder(tf.int32, shape=(cfg.batch_size))
    batch_x_squash = tf.divide(batch_x, 255.)
    batch_x = slim.batch_norm(batch_x, center=False, is_training=True, trainable=True)
    output = build_cnn_baseline(batch_x, is_train=True, num_classes=num_classes)
    loss, recon_loss, _ = cross_ent_loss(output, batch_x_squash, batch_labels)
    acc = test_accuracy(output, batch_labels)
    
    tf.summary.scalar('train_acc', acc)
    tf.summary.scalar('recon_loss', recon_loss)
    tf.summary.scalar('all_loss', loss)

    """Compute gradient."""
    grad = opt.compute_gradients(loss)
    grad_check = [tf.check_numerics(g, message='Gradient NaN Found!')
                  for g, _ in grad if g is not None] + [tf.check_numerics(loss, message='Loss NaN Found')]

    """Apply gradient."""
    with tf.control_dependencies(grad_check):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.apply_gradients(grad, global_step=global_step)

    """ RUN GRAPH """
    with tf.Session() as sess:
        if cfg.phase == 'train':
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            tf.get_default_graph().finalize()

            """Set summary writer"""
            if not os.path.exists(cfg.logdir + '/cnn_baseline/{}/train_log/'.format(dataset_name)):
                os.makedirs(cfg.logdir + '/cnn_baseline/{}/train_log/'.format(dataset_name))
            summary_writer = tf.summary.FileWriter(
                cfg.logdir + '/cnn_baseline/{}/train_log/'.format(dataset_name), graph=sess.graph)

            """Main loop"""
            for e in tqdm(list(range(cfg.num_epochs)), desc='epoch'):
                for b in tqdm(list(range(dataset.num_batches)), desc='batch'):
                    print(e, b)
                    batch = dataset.next_batch()
                    feed_dict = {images: batch.X, batch_labels: batch.y}
                    _, loss_value, summary_str = sess.run([train_op, loss, summary_op], feed_dict=feed_dict)
                    train_writer.add_summary(summary, global_step)
                dataset.reset()

if __name__ == "__main__":
    tf.app.run()
