# import argparse

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.autograd import Variable
# from torch.utils.data import DataLoader
# from torch.utils.data import sampler

# import torchvision.datasets as dset
# import torchvision.transforms as T

# from models import resnet

# import numpy as np

# # from dataset import Dataset

# from utils_pyT import load_mscoco#, test_accuracy



# parser = argparse.ArgumentParser()
# parser.add_argument('--data-dir', default='./dataset', type=str,
#                     help='path to dataset')
# parser.add_argument('--weight-decay', default=0.0001, type=float,
#                     help='parameter to decay weights')
# parser.add_argument('--batch-size', default=128, type=int,
#                     help='size of each batch of cifar-10 training images')
# parser.add_argument('--print-every', default=100, type=int,
#                     help='number of iterations to wait before printing')
# parser.add_argument('-n', default=5, type=int,
#                     help='value of n to use for resnet configuration (see https://arxiv.org/pdf/1512.03385.pdf for details)')
# parser.add_argument('--use-dropout', default=False, const=True, nargs='?',
#                     help='whether to use dropout in network')
# parser.add_argument('--res-option', default='A', type=str,
#                     help='which projection method to use for changing number of channels in residual connections')

# def main(args):
    

#     # # trainset.X has the RGB image arrays 
#     # X = trainset.X
#     # # trainset.y has the category ids    
#     # y = trainset.y
#     # trainset = Dataset("data/train/images/train2014")

#     """ GET DATA """
#     dataset = load_mscoco(cfg.phase, cfg, return_dataset=True)
#     N, D = dataset.X.shape
#     num_classes = 91
#     print("\nNum classes", num_classes)
#     num_batches_per_epoch = int(N / cfg.batch_size)

#     print("\nNum batches per epoch", num_batches_per_epoch)
#     print("\n, \n = ", (N, D))


#     print("END OF GET DATA")

# #     """ SET UP INITIAL VARIABLES"""
# #     learning_rate = tf.constant(cfg.initial_learning_rate)
# #     opt = tf.train.AdamOptimizer(cfg.initial_learning_rate)
# #     global_step = tf.Variable(0, name='global_step', trainable=False)
# #     # tf.summary.scalar('learning_rate', lrn_rate)

# #     """ DEFINE DATA FLOW """
# #     batch_x = tf.placeholder(tf.float32, shape=(cfg.batch_size, D, D, 3), name="input")
# #     batch_labels = tf.placeholder(tf.int32, shape=(cfg.batch_size), name="labels")
# #     batch_x_squash = tf.divide(batch_x, 255.)
# #     batch_x_norm = slim.batch_norm(batch_x, center=False, is_training=True, trainable=True)
# #     output = build_cnn_baseline(batch_x_norm, is_train=True, num_classes=num_classes)
# #     loss, recon_loss, _ = cross_ent_loss(output, batch_x_squash, batch_labels)
# #     acc = test_accuracy(output, batch_labels)
    
# #     tf.summary.scalar('train_acc', acc)
# #     tf.summary.scalar('recon_loss', recon_loss)
# #     tf.summary.scalar('all_loss', loss)

# #     # define transforms for normalization and data augmentation
# #     transform_augment = T.Compose([
# #         T.RandomHorizontalFlip(),
# #         T.RandomCrop(32, padding=4)])
# #     transform_normalize = T.Compose([
# #         T.ToTensor(),
# #         T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# #     ])
# #     # get MSCOCO Data

# #     # FIX NUM_TRAIN AND NUM_VAL
# #     NUM_TRAIN = 82783
# #     NUM_VAL = 40504
# #     print('REACHING')
# #     coco_train = dset.coco('./dataset', train=True, download=True,
# #                                  transform=T.Compose([transform_augment, transform_normalize]))
# #     loader_train = DataLoader(coco_train, batch_size=args.batch_size,
# #                               sampler=ChunkSampler(NUM_TRAIN))
# #     print('REACHING')

# #     coco_val = dset.CIFAR10('./dataset', train=True, download=True,
# #                                transform=transform_normalize)
# #     loader_val = DataLoader(coco_train, batch_size=args.batch_size,
# #                             sampler=ChunkSampler(NUM_VAL, start=NUM_TRAIN))
# #     coco_test = dset.coco('./dataset', train=False, download=True,
# #                                 transform=transform_normalize)
# #     loader_test = DataLoader(coco_test, batch_size=args.batch_size)
    




# #     # load model
# #     model = ResNet(args.n, res_option=args.res_option, use_dropout=args.use_dropout)
    
# #     param_count = get_param_count(model)
# #     print('Parameter count: %d' % param_count)
    
# #     # use gpu for training
# #     if not torch.cuda.is_available():
# #         print('Error: CUDA library unavailable on system')
# #         return
# #     global gpu_dtype
# #     gpu_dtype = torch.cuda.FloatTensor
# #     model = model.type(gpu_dtype)
    
# #     # setup loss function
# #     criterion = nn.CrossEntropyLoss().cuda()
# #     # train model
# #     SCHEDULE_EPOCHS = [50, 5, 5] # divide lr by 10 after each number of epochs
# # #     SCHEDULE_EPOCHS = [100, 50, 50] # divide lr by 10 after each number of epochs
# #     learning_rate = 0.1
# #     for num_epochs in SCHEDULE_EPOCHS:
# #         print('Training for %d epochs with learning rate %f' % (num_epochs, learning_rate))
# #         optimizer = optim.SGD(model.parameters(), lr=learning_rate,
# #                               momentum=0.9, weight_decay=args.weight_decay)
# #         for epoch in range(num_epochs):
# #             check_accuracy(model, loader_val)
# #             print('Starting epoch %d / %d' % (epoch+1, num_epochs))
# #             train(loader_train, model, criterion, optimizer)
# #         learning_rate *= 0.1
    
# #     print('Final test accuracy:')
# #     check_accuracy(model, loader_test)

# def check_accuracy(model, loader):
#     num_correct = 0
#     num_samples = 0
#     model.eval()
#     for X, y in loader:
#         X_var = Variable(X.type(gpu_dtype), volatile=True)

#         scores = model(X_var)
#         _, preds = scores.data.cpu().max(1)

#         num_correct += (preds == y).sum()
#         num_samples += preds.size(0)

#     acc = float(num_correct) / num_samples
#     print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

# def train(loader_train, model, criterion, optimizer):
#     model.train()
#     for t, (X, y) in enumerate(loader_train):
#         X_var = Variable(X.type(gpu_dtype))
#         y_var = Variable(y.type(gpu_dtype)).long()

#         scores = model(X_var)

#         loss = criterion(scores, y_var)
#         if (t+1) % args.print_every == 0:
#             print('t = %d, loss = %.4f' % (t+1, loss.data[0]))

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

# def get_param_count(model):
#     param_counts = [np.prod(p.size()) for p in model.parameters()]
#     return sum(param_counts)

# class ChunkSampler(sampler.Sampler):
#     def __init__(self, num_samples, start=0):
#         self.num_samples = num_samples
#         self.start = start
    
#     def __iter__(self):
#         return iter(range(self.start, self.start+self.num_samples))
    
#     def __len__(self):
#         return self.num_samples

# if __name__ == '__main__':
#     args = parser.parse_args()
#     main(args)



import tensorflow as tf
import tensorflow.contrib.slim.nets as resnet_v1
from config import cfg
from utils import create_inputs_mscoco, load_mscoco, test_accuracy
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
    N, D = dataset.X.shape
    num_classes = 91
    print("\nNum classes", num_classes)
    num_batches_per_epoch = int(N / cfg.batch_size)

    print("\nNum batches per epoch", num_batches_per_epoch)
    print("\n, \n = ", (N, D))


    print("END OF GET DATA")


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
