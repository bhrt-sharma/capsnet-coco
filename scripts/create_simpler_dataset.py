import os
import json
import numpy as np
from tqdm import tqdm
from scipy.ndimage import imread
import pdb

ORIGIN_FOLDER = 'data/train/images/masked/'
TARGET_CATEGORIES = [1, 62] # 1 is person, 62 is chair, 3 is car. 17 is cat, 18 is dog
TARGET_NUMBER_OF_IMAGES_PER_CLASS = {
	'train': 5000,
	'val': 1000,
	'test': 1000
}

TRAIN_FOLDER = 'data/train/images/simple-2/'
VAL_FOLDER = 'data/val/images/simple-2/'
TEST_FOLDER = 'data/test/images/simple-2/'

def copy_good_images_over():
	with open("data/good_train_test_val_split.json", "r") as f:
		dataset_split = json.load(f)

	train_files = []
	val_files = []
	test_files = []

	for category in TARGET_CATEGORIES:
		train_files = train_files + dataset_split['train'][str(category)]
		val_files = val_files + dataset_split['val'][str(category)]
		test_files = test_files + dataset_split['test'][str(category)]

	for f_name in tqdm(train_files, desc="copying train files"):
		os.system("cp {}/{} {}".format(ORIGIN_FOLDER, f_name, TRAIN_FOLDER))

	for f_name in tqdm(val_files, desc="copying val files"):
		os.system("cp {}/{} {}".format(ORIGIN_FOLDER, f_name, VAL_FOLDER))

	for f_name in tqdm(test_files, desc="copying test files"):
		os.system("cp {}/{} {}".format(ORIGIN_FOLDER, f_name, TEST_FOLDER))

# defined by the variance of color pixels
def is_good_image(image):
	std = np.std(image)
	return std > 5 # lol 

def find_good_images():
	image_files = os.listdir(ORIGIN_FOLDER)
	image_files = [f for f in image_files if ".jpg" in f]

	images_to_keep = {
		'train': {category_id: [] for category_id in TARGET_CATEGORIES},
		'val': {category_id: [] for category_id in TARGET_CATEGORIES},
		'test': {category_id: [] for category_id in TARGET_CATEGORIES},
	}

	for category in TARGET_CATEGORIES:
		print("Looking through photos of category %d" % category)
		category_image_files = [f for f in image_files if int(f.split("_")[-2]) == category]
		np.random.shuffle(category_image_files)

		for dataset in TARGET_NUMBER_OF_IMAGES_PER_CLASS:
			print("Finding %s files" % dataset)
			target_num = TARGET_NUMBER_OF_IMAGES_PER_CLASS[dataset]

			num_images_obtained = 0
			while category_image_files and num_images_obtained < target_num:
				img_file = category_image_files.pop(0)
				I = imread(ORIGIN_FOLDER + img_file, mode="L")
				if is_good_image(I):
					if (num_images_obtained % 500) == 0:
						print("Obtained %d good images for %s set" % (num_images_obtained, dataset))
					images_to_keep[dataset][category].append(img_file)
					num_images_obtained += 1

	with open("good_train_test_val_split.json", "w+") as f:
		json.dump(images_to_keep, f)

if __name__ == '__main__':
	# find_good_images()
	copy_good_images_over()
