'''
ASSUMPTIONS

-images have already been preprocessed according to README file @ https://github.com/handrew/capsnet-coco
- 


DON'T FORGET TO DOWNSAMPLE IMAGES! The images are too large as they stand, so in order to make this tractable we should probably downsample to 64x64 or 128x128.

Modify Dataset depending on how you guys want to load batches.
Baseline: train a resnet model on the raw data, test on raw data.
Baseline on masked: train a resnet model on the masked images, test on masked images.
Baseline on masked, detexturized: train a resnet model on the detexturized, masked images, test on detexturized, masked images.
Capsnet: train a capsnet model on the raw data, test on raw data.
Capsnet on masked: train a capsnet model on the masked images, test on masked images.
Baseline on masked, detexturized: train a capsnet model on the detexturized, masked images, test on detexturized, masked images.

CITATIONS

-ResNet based on this GitHub pytorch implementation: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as transformations

import numpy as np

from utils.dataset import Dataset


def main():
	# Load images by instantiating a Dataset object.

	# train_set.X has the RGB image arrays 
	# train_set.y has the category ids
	train_set = Dataset("data/train/images/train2014")
	# val_set.X has the RGB image arrays 
	# val_set.y has the category ids
	val_set = Dataset("data/val/images/val2014")
	# test_set.X has the RGB image arrays 
	# test_set.y has the category ids
	test_set = Dataset("data/test/images/test2014")


	# MODULE API and SEQUENTIAL API used to implement ResNet

	# 34 layer residual 

	'''
	When we use fully connected affine layers to process the image, however, we want each datapoint to be represented by a single vector -- it's no longer useful to segregate the different channels, rows, and columns of the data. So, we use a "flatten" operation to collapse the C x H x W values per representation into a single long vector. The flatten function below first reads in the N, C, H, and W values from a given batch of data, and then returns a "view" of that data. "View" is analogous to numpy's "reshape" method: it reshapes x's dimensions to be N x ??, where ?? is allowed to be anything (in this case, it will be C x H x W, but we don't need to specify that explicitly).
	'''

	def flatten(x):
	    N = x.shape[0] # read in N, C, H, W
	    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image



	# PERFORM RELEVANT PREPROCESSING


	# LOOK INTO WHAT IN_PLANES ARE AND OUT_PLANES


if __name__ == '__main__':
	main()
