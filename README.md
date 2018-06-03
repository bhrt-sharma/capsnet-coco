# Setup

It seems that the image provided does not have the Python Imaging Library installed. It does, however, have Anaconda installed. Make sure that you have followed the instructions on the tutorial and 

`/home/shared/setup.sh && source ~/.bashrc` from the home directory. 

Then run `conda install pillow`. 

Then, we need to build the pycoco tools first. Navigate to data/cocoapi/pycoco and run 

`python setup.py build_ext --inplace`

`rm -rf build`

# Download 

## MSCOCO

To download all of the MSCOCO datasets (i.e., train, val, and test), simply run `sh download_mscoco.sh`. 

This will, in order:

1. Run the python script to download CoCo images (download zip files, unzip them into right directories).
2. Remove the zip files. 
3. Download the train / val captions, unzip them, move them to the right directories, and remove the unzipped folder. 
4. Download the test captions, unzip them, move to the right directory, and remove the unzipped folder. 
5. Remove all zip files for space purposes. 

See the flags within `data/download_coco.py` for further details if you want to redownload one specific dataset. 

## t-less

Similarly:

`sh download_tless.sh`

# Preprocessing (Masking) MSCOCO

To obtain the ground truth masks, simply:

`python data/cocoapi/pycoco/mask_raw_images.py -ds train -ps -sh`

Note that the `-ps` and `-sh` flags pad images to a square size and shrink them to 48x48, respectively. 

Images are named something like: "COCO_train2014_000000057870.jpg". The number at the end is the id. `mask_raw_images` will output masked images to, for example, `data/train/images/masked`, with a slightly amended file name, such as `COCO_train2014_000000057870_18_2.jpg`. The extra numbers at the end is meant to denote the category id and the number of times we saw that category. In the previous example, we saw category 18 twice. We can then obtain the category itself by instantiating a COCO object and calling `coco.loadCats([ids])`. Category id 18 above for example, will return something like 

`[{u'supercategory': u'animal', u'id': 18, u'name': u'dog'}]`

Finally, run 

`python scripts/create_train_val_test_set.py` 

# Acknowledgements

We used [this repository's implementation](https://github.com/gyang274/capsulesEM) of capsule networks with EM routing.