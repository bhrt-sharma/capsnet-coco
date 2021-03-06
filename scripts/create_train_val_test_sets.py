import os
from tqdm import tqdm
import json
import numpy as np
import pdb

ORIGIN_FOLDER = 'data/train/images/masked/'
TRAIN_FOLDER = 'data/train/images/final/'
VAL_FOLDER = 'data/val/images/final/'
TEST_FOLDER = 'data/test/images/final/'

DIRECTORY_JSON = 'data/train_test_val_split.json'


def copy_files_over():
    with open(DIRECTORY_JSON, "r") as f:
        dataset_split = json.load(f)
    train_files = dataset_split['train']
    val_files = dataset_split['val']
    test_files = dataset_split['test']

    for f_name in tqdm(train_files, desc="copying train files"):
        os.system("cp {}/{} {}".format(ORIGIN_FOLDER, f_name, TRAIN_FOLDER))

    for f_name in tqdm(val_files, desc="copying val files"):
        os.system("cp {}/{} {}".format(ORIGIN_FOLDER, f_name, VAL_FOLDER))

    for f_name in tqdm(test_files, desc="copying test files"):
        os.system("cp {}/{} {}".format(ORIGIN_FOLDER, f_name, TEST_FOLDER))


def get_train_test_val_info():
    train_files = os.listdir(TRAIN_FOLDER)
    val_files = os.listdir(VAL_FOLDER)
    test_files = os.listdir(TEST_FOLDER)

    train_files = [f for f in train_files if '.jpg' in f]
    val_files = [f for f in val_files if '.jpg' in f]
    test_files = [f for f in test_files if '.jpg' in f]

    print("Dumping.")
    file_names = {"train": train_files, "val": val_files, "test": test_files}
    with open(DIRECTORY_JSON, "w+") as f:
        json.dump(file_names, f)


def shuffle():
    num_train = 80000
    num_val = 40000
    num_test = 40000

    # get files and counts of categories
    masked_files = os.listdir(ORIGIN_FOLDER)
    masked_files = [f for f in masked_files if '.jpg' in f]

    print("There are {} masked files.".format(len(masked_files)))

    category_dict = {}
    for f in tqdm(masked_files, desc="loading to dict"):
        cat_id = f.split("_")[-2]
        img = f
        category_dict[cat_id] = category_dict.get(cat_id, []) + [img]

    num_categories = len(category_dict.keys())

    print("There are {} categories.".format(num_categories))

    # choose files for train test and val 
    for category in tqdm(category_dict, desc="category"):
        images = category_dict[category]
        np.random.shuffle(images)

        num_train_files_to_get = num_train // num_categories
        num_val_files_to_get = num_val // num_categories
        num_test_files_to_get = num_test // num_categories

        total_num = num_train_files_to_get + num_val_files_to_get + num_test_files_to_get
        if total_num >= len(images):
            num_train_files_to_get = int(len(images) * num_train_files_to_get / total_num)
            num_val_files_to_get = int(len(images) * num_val_files_to_get / total_num)
            num_test_files_to_get = int(len(images) * num_test_files_to_get / total_num)

        print("Getting {} train files for category {}.".format(num_train_files_to_get, category))
        for tr in range(num_train_files_to_get):
            f_name = images.pop(0)
            os.system("cp {}/{} {}".format(ORIGIN_FOLDER, f_name, TRAIN_FOLDER))

        print("Getting {} val files for category {}.".format(num_val_files_to_get, category))
        for v in range(num_val_files_to_get):
            f_name = images.pop(0)
            os.system("cp {}/{} {}".format(ORIGIN_FOLDER, f_name, VAL_FOLDER))

        print("Getting {} test files for category {}.".format(num_test_files_to_get, category))
        for te in range(num_test_files_to_get):
            f_name = images.pop(0)
            os.system("cp {}/{} {}".format(ORIGIN_FOLDER, f_name, TEST_FOLDER))


if __name__ == '__main__':
    # shuffle()
    # get_train_test_val_info()
    copy_files_over()
