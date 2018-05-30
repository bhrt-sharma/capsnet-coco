import tensorflow as tf
import tensorflow.contrib.slim as slim
from config import cfg
from utils import load_mscoco, test_accuracy, one_hot_encode
import numpy as np
import os
from tqdm import tqdm
from models.capsules import nets


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  num_classes = 91

  print("\nGetting train data...")
  train_dataset = load_mscoco('train', cfg, return_dataset=True)
  print("\nGetting val data...")
  val_dataset = load_mscoco('val', cfg, return_dataset=True)

  num_examples = train_dataset.X.shape[0]
  num_steps_per_epoch = int(num_examples / cfg.batch_size)

  with tf.Graph().as_default():
    with tf.device('/cpu:0'):
      global_step = tf.train.get_or_create_global_step()

    images, labels = train_dataset.X.astype(np.float32), train_dataset.y
    one_hot_labels = one_hot_encode(labels, num_classes)

    # create batches
    data_queues = tf.train.slice_input_producer([images, one_hot_labels, labels])
    images, one_hot_labels, labels = tf.train.shuffle_batch(
      data_queues,
      num_threads=16,
      batch_size=cfg.batch_size,
      capacity=cfg.batch_size * 64,
      min_after_dequeue=cfg.batch_size * 32,
      allow_smaller_final_batch=False)

    poses, activations = nets.capsules_v0(images, num_classes=num_classes, iterations=cfg.iter_routing, cfg=cfg, name='capsulesEM-V0')

    train_accuracy = test_accuracy(activations, labels)

    # margin schedule
    # margin increase from 0.2 to 0.9 after margin_schedule_epoch_achieve_max
    margin_schedule_epoch_achieve_max = 10.0
    margin = tf.train.piecewise_constant(
      tf.cast(global_step, dtype=tf.int32),
      boundaries=[
        int(num_steps_per_epoch * margin_schedule_epoch_achieve_max * x / 7) for x in range(1, 8)
      ],
      values=[
        x / 10.0 for x in range(2, 10)
      ]
    )

    loss = nets.spread_loss(
      one_hot_labels, activations, margin=margin, name='spread_loss'
    )

    tf.summary.scalar('losses/spread_loss', loss)
    
    # exponential learning rate decay
    learning_rate = tf.train.exponential_decay(
      cfg.initial_learning_rate,
      global_step,
      decay_steps = num_steps_per_epoch,
      decay_rate = 0.8,
      staircase = True)

    optimizer = tf.train.AdamOptimizer(learning_rate=cfg.initial_learning_rate)

    train_tensor = slim.learning.create_train_op(
      loss, optimizer, global_step=global_step, clip_gradient_norm=4.0
    )

    print("\nTraining... Learning rate: %0.5f\n" % cfg.initial_learning_rate)

    def train_step_fn(session, *args, **kwargs):
      total_loss, should_stop = slim.learning.train_step(session, *args, **kwargs)

      def get_accuracy_for_dataset(dset):
        print("Getting accuracy... ")
        num_batches_in_train = 0
        mean_acc = 0.0
        while dset.has_next_batch():
          num_batches_in_train += 1
          curr_X, curr_labels = dset.next_batch()
          curr_X = curr_X.astype(np.float32)
          curr_train_acc, step_out = session.run([train_accuracy, global_step], feed_dict={images: curr_X, labels: curr_labels})
          mean_acc += curr_train_acc
        mean_acc = mean_acc / num_batches_in_train
        dset.reset()
        return mean_acc

      def write_summary(tag, value, step_out):
        sum_writer = tf.summary.FileWriter(cfg.logdir, graph=session.graph)
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        sum_writer.add_summary(summary, step_out)

      if (train_step_fn.step % 100 == 0):
        mean_train_acc = get_accuracy_for_dataset(train_dataset)
        mean_val_acc = get_accuracy_for_dataset(val_dataset)
        write_summary('accuracies/train_acc', mean_train_acc, step_out)
        write_summary('accuracies/val_acc', mean_val_acc, step_out)
        print('Step %s - Train Accuracy: %.2f%% - Val Accuracy: %.2f%%' % (str(train_step_fn.step).rjust(6, '0'), mean_train_acc, mean_val_acc))

      train_step_fn.step += 1
      return [total_loss, should_stop]

    train_step_fn.step = 0

    slim.learning.train(
      train_tensor,
      logdir=cfg.logdir,
      log_every_n_steps=10,
      save_summaries_secs=60 * 5,
      saver=tf.train.Saver(max_to_keep=100),
      save_interval_secs=600,
      # yg: add session_config to limit gpu usage and allow growth
      session_config=tf.ConfigProto(
        # device_count = {
        #   'GPU': 0
        # },
        gpu_options={
          'allow_growth': 0,
          # 'per_process_gpu_memory_fraction': 0.01
          'visible_device_list': '0'
        },
        allow_soft_placement=True,
        log_device_placement=False,
      ),
      train_step_fn=train_step_fn
    )

if __name__ == "__main__":
  tf.app.run()
