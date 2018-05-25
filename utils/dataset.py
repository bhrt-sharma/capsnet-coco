import os
import math
import numpy as np
import skimage.io as io


class Dataset(object):
    def __init__(self, 
                 folder_name, 
                 batch_size=64, 
                 is_train=False, 
                 num=None,
                 shuffle=True):
        self.folder_name = folder_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        image_files = os.listdir(folder_name)
        image_files = [f for f in image_files if ".jpg" in f]
        if type(num) == int:
            image_files = [image_files[f] for f in range(num)]

        self.X = []
        self.y = []

        for img_file in image_files:
            img_arr = io.imread("{}/{}".format(folder_name, img_file))

            file_parts = img_file.split("_")
            instance_seen = int(file_parts[-1].replace(".jpg", ""))
            category_id = int(file_parts[-2])
            
            self.X.append(img_arr)
            self.y.append(category_id)

        self.X = np.asarray(self.X)
        self.y = np.asarray(self.y)

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

         imgs = self.X[current_idxs]
         if self.is_train:
             categories = self.y[current_idxs]
             self.current_idx += self.batch_size
             return image_files, word_idxs, masks
         else:
             self.current_idx += self.batch_size
             return image_files

    def has_next_batch(self):
         """ Determine whether there is a batch left. """
         return self.current_idx < self.count

    def has_full_next_batch(self):
         """ Determine whether there is a full batch left. """
         return self.current_idx + self.batch_size <= self.count
