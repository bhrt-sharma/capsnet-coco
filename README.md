All right boys let's do this.

# TODO

## Preprocessing

1. Write a dataset object... 
2. Modify `getGroundTruthMasks` in coco.py to also return non-polygon segments
3. Add neutral grey background (instead of the black background? Not sure if this should theoretically make any difference).
4. Figure out a detexturizing algorithm – either reduce contrast or add a small amount of gaussian blur. 

## Training and such

DON'T FORGET TO DOWNSAMPLE IMAGES! The images are too large as they stand, so in order to make this tractable we should probably downsample to 64x64 or 128x128. 

1. Baseline: train a resnet model on the raw data, test on raw data. 
2. Baseline on masked: train a resnet model on the masked images, test on masked images.
3. Baseline on masked, detexturized: train a resnet model on the detexturized, masked images, test on detexturized, masked images.
4. Capsnet: train a capsnet model on the raw data, test on raw data. 
5. Capsnet on masked: train a capsnet model on the masked images, test on masked images.
6. Baseline on masked, detexturized: train a capsnet model on the detexturized, masked images, test on detexturized, masked images.

# Download 

To download all of the datasets (i.e., train, val, and test), simply run `sh download_dataset.sh`. 

This will, in order:

1. Run the python script to download CoCo images (download zip files, unzip them into right directories).
2. Remove the zip files. 
3. Download the train / val captions, unzip them, move them to the right directories, and remove the unzipped folder. 
4. Download the test captions, unzip them, move to the right directory, and remove the unzipped folder. 
5. Remove all zip files for space purposes. 

See the flags within `data/download_coco.py` for further details if you want to redownload one specific dataset. 

# Setup

It seems that the image provided does not have the Python Imaging Library installed. It does, however, have Anaconda installed. Make sure that you have followed the instructions on the tutorial and 

`/home/shared/setup.sh && source ~/.bashrc` from the home directory. 

Then run `conda install pillow`. 

Then, we need to build the pycoco tools first. Navigate to data/cocoapi/pycoco and run 

`python setup.py build_ext --inplace`

`rm -rf build`

# Preprocessing (Masking)

To obtain the ground truth masks, simply:

`python data/cocoapi/pycoco/pycocotools/mask_raw_images.py -ds train` 

`python data/cocoapi/pycoco/pycocotools/mask_raw_images.py -ds val` 

`python data/cocoapi/pycoco/pycocotools/mask_raw_images.py -ds test` 


Optionally add the `-rt` flag in order to remove textures from images too. *This has not yet been implemented.*


Images are named something like: "COCO_train2014_000000057870.jpg". The number at the end is the id. `mask_raw_images` will output masked images to, for example, `data/train/images/masked`, with a slightly amended file name, such as "COCO_train2014_000000057870_18.jpg". The extra number at the end is meant to denote the category id. We can then obtain the category itself by instantiating a COCO object and calling `coco.loadCats([ids])`. Category id 18 above for example, will return something like 

`[{u'supercategory': u'animal', u'id': 18, u'name': u'dog'}]`