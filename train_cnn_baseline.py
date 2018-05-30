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

    assert len(args) == 2 and isinstance(args[1], str)
    experiment_name = args[1]
    """ GET DATA """
    train_dataset = load_mscoco(cfg.phase, cfg, return_dataset=True)
    val_dataset = load_mscoco('val', cfg, return_dataset=True)
    dataset_name = 'mscoco' 
    checkpoints_to_keep = 1
    N, D = train_dataset.X.shape[0], train_dataset.X.shape[1]
    num_classes = 91
    print("\nNum classes", num_classes)
    num_batches_per_epoch = int(N / cfg.batch_size)

    """ SET UP INITIAL VARIABLES"""
    learning_rate = tf.constant(cfg.initial_learning_rate)
    opt = tf.train.AdamOptimizer(cfg.initial_learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    tf.summary.scalar('learning_rate', learning_rate)

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

    # set best checkpoint
    bestmodel_dir = os.path.join(cfg.logdir + '/cnn_baseline/{}'.format(experiment_name), 'best_checkpoint')
    if not os.path.exists(bestmodel_dir):
        os.makedirs(bestmodel_dir)
    bestmodel_ckpt_path = os.path.join(bestmodel_dir, "cnn_best.ckpt")
    train_saver = tf.train.Saver(tf.global_variables(), max_to_keep = checkpoints_to_keep)
    bestmodel_saver = tf.train.Saver(tf.global_variables(), max_to_keep = checkpoints_to_keep)
    summary_op = tf.summary.merge_all()

    """ RUN GRAPH """
    with tf.Session() as sess:
        if cfg.phase == 'train':
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            tf.get_default_graph().finalize()

            """Set summary writers"""
            if not os.path.exists(cfg.logdir + '/cnn_baseline/{}/train_log/'.format(experiment_name)):
                os.makedirs(cfg.logdir + '/cnn_baseline/{}/train_log/'.format(experiment_name))
            summary_writer = tf.summary.FileWriter(
                cfg.logdir + '/cnn_baseline/{}/train_log/'.format(experiment_name), graph=sess.graph)

            # if not os.path.exists(cfg.logdir + '/cnn_baseline/{}_images/val_log/'.format(cfg.phase)):
            #     os.makedirs(cfg.logdir + '/cnn_baseline/{}_images/val_log/'.format(cfg.phase))
            # val_writer = tf.summary.FileWriter(
            #     cfg.logdir + '/cnn_baseline/{}_images/val_log/'.format(cfg.phase), graph=sess.graph)
            """Main loop"""
            best_loss = None
            best_acc = None 
            for e in list(range(cfg.num_epochs)):
                for b in list(range(num_batches_per_epoch)):
                    batch = train_dataset.next_batch()
                    feed_dict = {batch_x: batch[0].astype(np.float32), batch_labels: batch[1]}
                    _, loss_value, accuracy, summary_str, step_out = sess.run([opt_op, loss, acc, summary_op, global_step], feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step_out)
                print("Loss and accuracy: ", loss_value, accuracy)
                train_dataset.reset()

                #save model after every epoch 
                print('saving model now :)')
                train_ckpt_path = os.path.join(
                    cfg.logdir + '/cnn_baseline/{}'.format(experiment_name), 'model-{:.4f}.ckpt'.format(loss_value))
                train_saver.save(sess, train_ckpt_path, global_step=step_out)

                #eval on validation loss 
                loss_per_val_batch = 0.0
                acc_per_val_batch = 0.0
                num_val_batches = 0
                while val_dataset.has_next_batch():
                    val_batch = val_dataset.next_batch()
                    feed_dict = {batch_x: val_batch[0].astype(np.float32), batch_labels: val_batch[1]}
                    dev_loss, dev_acc, summary_str, step_out = sess.run([loss, acc, summary_op, global_step], feed_dict=feed_dict)
                    loss_per_val_batch += loss_value
                    acc_per_val_batch += dev_acc
                    num_val_batches +=1 
                avg_val_loss = loss_per_val_batch / num_val_batches
                avg_val_acc = acc_per_val_batch / num_val_batches
                write_summary(avg_val_loss, "dev/avg_validation_loss", summary_writer, step_out)
                write_summary(avg_val_acc, "dev/avg_validation_acc", summary_writer, step_out)

                if best_acc is None or avg_val_acc > best_acc:
                    print('saving best model!')
                    bestmodel_saver.save(sess, bestmodel_ckpt_path, global_step=step_out)
                    best_acc = avg_val_acc
                val_dataset.reset()

def write_summary(value, tag, summary_writer, global_step):
    """Write a single summary value to tensorboard"""
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)

if __name__ == "__main__":
    tf.app.run()
