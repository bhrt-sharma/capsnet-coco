All right boys let's do this.

# Download 

To download all of the datasets (i.e., train, val, and test), simply run `sh download_dataset.sh`. 

This will, in order:

1. Run the python script to download CoCo images (download zip files, unzip them into right directories).
2. Remove the zip files. 
3. Download the train / val captions, unzip them, move them to the right directories, and remove the unzipped folder. 
4. Download the test captions, unzip them, move to the right directory, and remove the unzipped folder. 
5. Remove all zip files for space purposes. 

See the flags within `data/download_coco.py` for further details if you want to redownload one specific dataset. 