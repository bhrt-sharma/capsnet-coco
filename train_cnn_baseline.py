import tensorflow as tf
import tensorflow.contrib.slim as slim
from config import cfg
from utils import load_mscoco, test_accuracy
import time
import numpy as np
import sys
import os
from tqdm import tqdm
from models.cnn_baseline import build_cnn_baseline, cross_ent_loss


def main(args):
    tf.set_random_seed(1234)

    """ GET DATA """
    dataset = load_mscoco(cfg.phase, cfg, return_dataset=True)
    N, D = dataset.X.shape[0], dataset.X.shape[1]
    num_classes = 91
    print("\nNum classes", num_classes)
    num_batches_per_epoch = int(N / cfg.batch_size)

    """ SET UP INITIAL VARIABLES"""
    learning_rate = tf.constant(cfg.initial_learning_rate)
    opt = tf.train.AdamOptimizer(cfg.initial_learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # tf.summary.scalar('learning_rate', lrn_rate)

    """ DEFINE DATA FLOW """
    batch_x = tf.placeholder(tf.float32, shape=(cfg.batch_size, D, D, 3), name="input")
    batch_labels = tf.placeholder(tf.int32, shape=(cfg.batch_size), name="labels")
    batch_x_squash = tf.divide(batch_x, 255.)
    batch_x_norm = slim.batch_norm(batch_x, center=False, is_training=True, trainable=True)
    output = build_cnn_baseline(batch_x_norm, is_train=True, num_classes=num_classes)
    loss, recon_loss, _ = cross_ent_loss(output, batch_x_squash, batch_labels)
    acc = test_accuracy(output, batch_labels)
    
    tf.summary.scalar('train_acc', acc)
    tf.summary.scalar('recon_loss', recon_loss)
    tf.summary.scalar('all_loss', loss)

    """Compute gradient."""
    def _learning_rate_decay_fn(learning_rate, global_step):
        return tf.train.exponential_decay(
                        learning_rate,
                        global_step,
                        decay_steps = num_batches_per_epoch,
                        decay_rate = 0.8,
                        staircase = True)

    opt_op = tf.contrib.layers.optimize_loss(
                loss = loss,
                global_step = global_step,
                learning_rate = learning_rate,
                optimizer = opt,
                # clip_gradients = False,
                learning_rate_decay_fn = _learning_rate_decay_fn)

    summary_op = tf.summary.merge_all()

    """ RUN GRAPH """
    with tf.Session() as sess:
        if cfg.phase == 'train':
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            tf.get_default_graph().finalize()

            """Set summary writer"""
            if not os.path.exists(cfg.logdir + '/cnn_baseline/{}_images/train_log/'.format(cfg.phase)):
                os.makedirs(cfg.logdir + '/cnn_baseline/{}_images/train_log/'.format(cfg.phase))
            summary_writer = tf.summary.FileWriter(
                cfg.logdir + '/cnn_baseline/{}_images/train_log/'.format(cfg.phase), graph=sess.graph)

            """Main loop"""
            for e in list(range(cfg.num_epochs)):
                for b in list(range(num_batches_per_epoch)):
                    batch = dataset.next_batch()
                    feed_dict = {batch_x: batch[0].astype(np.float32), batch_labels: batch[1]}
                    _, loss_value, accuracy, summary_str, step_out = sess.run([opt_op, loss, acc, summary_op, global_step], feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step_out)
                print("Loss and accuracy: ", loss_value, accuracy)
                dataset.reset()

if __name__ == "__main__":
    tf.app.run()
