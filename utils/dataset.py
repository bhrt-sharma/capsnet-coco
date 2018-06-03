import os
import math
import numpy as np
import skimage.io as io
from scipy.ndimage import imread
import scipy


def load_mscoco(dataset_type, config, num=None, return_dataset=False):
    if dataset_type == 'train':
        data = Dataset("data/train/images/simple-2", is_train=True, batch_size=config.batch_size, num=num, greyscale=config.greyscale)
    elif dataset_type == 'test':
        data = Dataset("data/test/images/simple-2", batch_size=config.batch_size, num=num, greyscale=config.greyscale)
    elif dataset_type == 'val':
        data = Dataset("data/val/images/simple-2", is_train=True, batch_size=config.batch_size, num=num, greyscale=config.greyscale)
    else:
        raise ValueError("Dataset type must be one of 'train', 'test', or 'val'")

    if return_dataset:
        return data

    return (data.X, data.y)


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
            if not greyscale:
                img_arr = io.imread("{}/{}".format(folder_name, img_file))
            else:
                img_arr = imread("{}/{}".format(folder_name, img_file), mode="L")[:,:,np.newaxis]

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


"""
Returns three TLessDataset objects, which have the same basic idea
as the Dataset objects 

Takes care of train/val/test split with the following algorithm:
(For each class...)
1. Shuffle a range from 0 to 1295, inclusive
2. Since each class has 1295 images, take the first 1036 images (~80%)
to be in the train set. 
3. Take 260/2 = 130 to be in the val and test splits, respectively. 
"""
def load_tless_split(config):
    train_proportion = 0.8
    val_proportion = 0.1
    test_proportion = 0.1

    num_images_per_item = 1296
    num_train_images = int(train_proportion * num_images_per_item)

    image_ids_per_class = np.arange(0, 1297)
    np.random.shuffle(image_ids_per_class)
    non_train_image_ids = image_ids_per_class[num_train_images : ]

    train_set = TLessDataset(
        image_ids_per_class[:num_train_images], 
        batch_size=config.batch_size, 
        greyscale=config.greyscale
    )
    val_set = TLessDataset(
        non_train_image_ids[ : len(non_train_image_ids)//2], 
        batch_size=config.batch_size, 
        greyscale=config.greyscale
    )
    test_set = TLessDataset(
        non_train_image_ids[len(non_train_image_ids)//2 : ], 
        batch_size=config.batch_size, 
        greyscale=config.greyscale
    )

    return train_set, val_set, test_set


class TLessDataset(Dataset):
    def __init__(self, files, batch_size=64, greyscale=False, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.current_idx = 0

        self.image_files = files

        all_image_folder = 'data/t-less/t-less_v2/train_primesense/'
        class_folders = os.listdir(all_image_folder)

        self.num_classes = len(class_folders)

        self.X = []
        self.y = []

        for class_folder in class_folders:
            img_folder = all_image_folder + class_folder + '/rgb/'

            class_images = os.listdir(img_folder)
            for img_file in class_images:
                curr_id = int(img_file.replace(".png", ""))
                if curr_id in self.image_files:
                    if not greyscale:
                        img_arr = io.imread("{}/{}".format(img_folder, img_file))
                    else:
                        img_arr = imread("{}/{}".format(img_folder, img_file), mode="L")[:,:,np.newaxis]

                    self.X.append(img_arr)
                    self.y.append(int(class_folder))

        self.setup()
