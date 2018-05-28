import os
from tqdm import tqdm
import numpy as np
import pdb

ORIGIN_FOLDER = 'data/train/images/masked/'
TRAIN_FOLDER = 'data/train/images/final/'
VAL_FOLDER = 'data/val/images/final/'
TEST_FOLDER = 'data/test/images/final/'


def copy_files_over():
	pass

def get_train_test_val():
	train_files = os.listdir(TRAIN_FOLDER)
	val_files = os.listdir(VAL_FOLDER)
	test_files = os.listdir(TEST_FOLDER)

	train_files = [f for f in train_files if '.jpg' in f]
	val_files = [f for f in val_files if '.jpg' in f]
	test_files = [f for f in test_files if '.jpg' in f]

	import json
	print("Dumping.")
	file_names = {"train": train_files, "val": val_files, "test": test_files}
	with open("train_test_val_split.json", "w+") as f:
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
		images = np.random.shuffle(category_dict[category])

		num_train_files_to_get = num_train / num_categories
		num_val_files_to_get = num_val / num_categories
		num_test_files_to_get = num_test / num_categories

		assert num_train_files_to_get + num_val_files_to_get + num_test_files_to_get < len(images)

		print("Getting train files for category {}.".format(category))
		for tr in tqdm(range(num_train_files_to_get)):
			f_name = images[tr]
			os.system("cp {}/{} {}".format(ORIGIN_FOLDER, f_name, TRAIN_FOLDER))
			images.remove(f_name)

		print("Getting val files for category {}.".format(category))
		for v in tqdm(range(num_val_files_to_get)):
			f_name = images[v]
			os.system("cp {}/{} {}".format(ORIGIN_FOLDER, f_name, VAL_FOLDER))
			images.remove(f_name)

		print("Getting test files for category {}.".format(category))
		for te in tqdm(range(num_test_files_to_get)):
			f_name = images[te]
			os.system("cp {}/{} {}".format(ORIGIN_FOLDER, f_name, TEST_FOLDER))
			images.remove(f_name)


if __name__ == '__main__':
	shuffle()
	get_train_test_val()