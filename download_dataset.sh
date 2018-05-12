echo "==========  Downloading train dataset"
python data/download_coco.py -ds train 
rm *_train.zip
echo "==========  Downloading val dataset."
python data/download_coco.py -ds val
rm *_val.zip
echo "==========  Downloading test dataset."
python data/download_coco.py -ds test
rm *_test.zip

echo "==========  Downloading train / val captions"
curl -O http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip -d .
mv annotations/captions_train2014.json data/train
mv annotations/captions_val2014.json data/val
mv annotations/instances_train2014.json data/train
mv annotations/instances_val2014.json data/val
rm -rf annotations/

echo "==========  Downloading test captions"
curl -O http://images.cocodataset.org/annotations/image_info_test2014.zip
unzip image_info_test2014.zip -d .
mv annotations/image_info_test2014.json data/test
rm -rf annotations/

echo "========== Cleaning up zip files"
rm *.zip
