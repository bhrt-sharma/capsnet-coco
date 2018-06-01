import os
import math
import numpy as np
import skimage.io as io
import scipy


class Dataset(object):
    def __init__(self, 
                 folder_name, 
                 batch_size=64, 
                 is_train=True, 
                 num=None,
                 greyscale=False,
                 shuffle=True):
        self.folder_name = folder_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.current_idx = 0
        self.is_train = is_train

        image_files = os.listdir(folder_name)
        self.image_files = [f for f in image_files if ".jpg" in f]
        if type(num) == int:
            self.image_files = [self.image_files[f] for f in range(num)]

        self.X = []
        self.y = []

        for img_file in self.image_files:
            img_arr = io.imread("{}/{}".format(folder_name, img_file), as_gray=greyscale)

            file_parts = img_file.split("_")
            instance_seen = int(file_parts[-1].replace(".jpg", ""))
            category_id = int(file_parts[-2])
            
            self.X.append(img_arr)
            self.y.append(category_id)

        self.X = np.asarray(self.X)
        self.y = np.asarray(self.y)

        self.setup()

    def setup(self):
        """ Setup the dataset. """
        self.count = len(self.image_files)
        self.num_batches = int(np.ceil(self.count * 1.0 / self.batch_size))
        self.fake_count = self.num_batches * self.batch_size - self.count
        self.idxs = list(range(self.count))
        self.reset()

    def next_batch(self):
         """ Fetch the next batch. """
         assert self.has_next_batch()

         if self.has_full_next_batch():
             start, end = self.current_idx, \
                          self.current_idx + self.batch_size
             current_idxs = self.idxs[start:end]
         else:
             start, end = self.current_idx, self.count
             current_idxs = self.idxs[start:end] + \
                            list(np.random.choice(self.count, self.fake_count))

         image_files = self.X[current_idxs]
         if self.is_train:
             categories = self.y[current_idxs]
             self.current_idx += self.batch_size
             return image_files, categories
         else:
             self.current_idx += self.batch_size
             return image_files

    def has_next_batch(self):
         """ Determine whether there is a batch left. """
         return self.current_idx < self.count

    def has_full_next_batch(self):
         """ Determine whether there is a full batch left. """
         return self.current_idx + self.batch_size <= self.count

    def reset(self):
        """ Reset the dataset. """
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.idxs)
