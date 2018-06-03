import os
import math
import numpy as np
import skimage.io as io
from skimage.transform import resize
from scipy.ndimage import imread
import json
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
def load_tless_split(config, num_classes=5):
    split_file = 'data/t-less_train_test_val_split.json'
    if os.path.isfile(split_file):
        with open(split_file, 'r') as f:
            split_dict = json.load(f)
            train_image_ids = split_dict['train']
            val_image_ids = split_dict['val']
            test_image_ids = split_dict['test']
    else:
        train_proportion = 0.8
        val_proportion = 0.1
        test_proportion = 0.1
        num_images_per_item = 1296
        num_train_images = int(train_proportion * num_images_per_item)
        
        image_ids_per_class = np.arange(0, 1297)
        np.random.shuffle(image_ids_per_class)

        train_image_ids = image_ids_per_class[:num_train_images]
        non_train_image_ids = image_ids_per_class[num_train_images : ]
        val_image_ids = non_train_image_ids[ : len(non_train_image_ids)//2]
        test_image_ids = non_train_image_ids[len(non_train_image_ids)//2 : ]

        with open(split_file, 'w+') as f:
            split_dict = {
                'train': train_image_ids.tolist(),
                'val': val_image_ids.tolist(),
                'test': test_image_ids.tolist()
            }
            json.dump(split_dict, f)

    train_set = TLessDataset(
        train_image_ids, 
        num_classes=num_classes,
        batch_size=config.batch_size, 
        greyscale=config.greyscale
    )
    val_set = TLessDataset(
        val_image_ids, 
        num_classes=num_classes,
        batch_size=config.batch_size, 
        greyscale=config.greyscale
    )
    test_set = TLessDataset(
        test_image_ids, 
        num_classes=num_classes,
        batch_size=config.batch_size, 
        is_train=False,
        greyscale=config.greyscale
    )

    return train_set, val_set, test_set


class TLessDataset(Dataset):
    def __init__(self, files, num_classes=5, batch_size=50, is_train=True, greyscale=False, shuffle=True, mean_center_unit_var=False):
        self.batch_size = batch_size
        self.is_train = is_train
        self.shuffle = shuffle
        self.current_idx = 0
        self.image_files = files
        self.greyscale = greyscale

        all_image_folder = 'data/t-less/t-less_v2/train_primesense/'
        class_folders = os.listdir(all_image_folder)

        self.X = []
        self.y = []
        self.included_class_folders = []

        for class_num in range(num_classes):
            class_folder = class_folders[class_num]
            self.included_class_folders.append(class_folder)

            img_folder = all_image_folder + class_folder + '/rgb/'

            class_images = os.listdir(img_folder)
            for img_file in class_images:
                curr_id = int(img_file.replace(".png", ""))
                if curr_id in self.image_files:
                    if not greyscale:
                        img_arr = io.imread("{}/{}".format(img_folder, img_file))
                        # two colors to mask out 
                        img_arr[np.where((img_arr<[30,30,50]).all(axis=2))] = [220, 220, 220]
                    else:
                        img_arr = imread("{}/{}".format(img_folder, img_file), mode="L")[:,:,np.newaxis]
                        img_arr[np.where((img_arr<=[50]))] = [220]
                    img_arr = self._crop_and_shrink_image(img_arr)
                    self.X.append(img_arr)
                    self.y.append(int(class_folder))

        self.X = np.asarray(self.X)
        self.y = np.asarray(self.y)

        self.setup()

    def _crop_and_shrink_image(self, im, crop_width=100, crop_height=100, new_width=48, new_height=48):
        width, height = im.shape[0], im.shape[1]  # Get dimensions

        left = (width - crop_width)//2
        top = (height - crop_height)//2
        right = (width + crop_width)//2
        bottom = (height + crop_height)//2

        cropped = im[top:bottom, left:right, :]

        resized = resize(cropped, (new_width, new_height))
        rescaled_image = 255 * resized
        final_image = rescaled_image.astype(np.uint8) # Convert to integer data type pixels.
    
        return final_image

    
