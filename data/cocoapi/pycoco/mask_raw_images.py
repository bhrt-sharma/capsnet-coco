import argparse
from PIL import Image, ImageDraw
import numpy as np
import os
from pycocotools.coco import COCO
from scipy.ndimage import imread, gaussian_filter


def getIdFromImage(file_name):
    return int(file_name.split("_")[-1].replace(".jpg", ""))

"""
@params:
- im is an **RGBA** Image as imported by scipy.ndimage.imread
- seg is segmentation array from the CoCoAPI
- out_name is what to call it 
- saveTo is a folder to save the image to 
- transparency determines whether the background is transparent (RGBA) or black (RGB)
"""
def maskSegmentOut(im, seg, background_image, out_name, saveTo=".."):
    # convert to numpy (for convenience)
    imArray = np.asarray(im)

    # create mask
    maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
    ImageDraw.Draw(maskIm).polygon(seg, outline=1, fill=1)
    mask = np.array(maskIm)

    # assemble new image (uint8: 0-255)
    newImArray = np.empty(imArray.shape,dtype='uint8')

    # colors (three first columns, RGB)
    newImArray[:,:,:3] = imArray[:,:,:3]

    # set transparency
    newImArray[:,:,3] = mask * 255

    # set colors to background image where transparency is 0
    newImArray[newImArray[:,:,3] == 0] = background_image[newImArray[:,:,3] == 0] 

    # gaussian filter 
    newImArray = gaussian_filter(newImArray, 0.9)

    # back to Image from numpy
    newIm = Image.fromarray(newImArray[:, :, :3], "RGB")

    newIm.save("{}/{}.jpg".format(saveTo, out_name))

def mask_all(args):
    # decide which folder to output to
    dataset = args.dataset
    out_folder = "data/{}/images/masked".format(dataset)

    # initialize coco API
    ann_file = 'data/{}/instances_{}2014.json'.format(dataset, dataset)
    cc = COCO(ann_file)

    # loop through folder to get the mean image
    # have to do a running mean because there are too many images 
    # to store in memory at once 
    raw_folder = "data/{}/images/{}2014".format(dataset, dataset)
    all_pics = os.listdir(raw_folder)
    all_pics = [pic for pic in all_pics if ".jpg" in pic]

    print("Computing mean image...")
    running_mean_image = imread("{}/{}".format(raw_folder, all_pics[0]), mode="RGB")
    num_images_processed = 1
    for i in range(1, len(all_pics)):
        pic = all_pics[i]
        I = imread("{}/{}".format(raw_folder, pic), mode="RGB")
        running_mean_image = running_mean_image * num_images_processed
        running_mean_image += I
        num_images_processed += 1
        running_mean_image = running_mean_image / num_images_processed

    # then loop through to actually perform the masking 
    # this is done in RGBA and not RGB
    for pic in all_pics:
        # get image 
        I = imread("{}/{}".format(raw_folder, pic), mode="RGBA")

        # converts something like "COCO_train2014_000000057870.jpg"
        # to 57870
        img_id = getIdFromImage(pic)

        # get annotation
        annIds = cc.getAnnIds(imgIds=img_id)
        anns = cc.loadAnns(annIds)

        # get segments from annotation
        segs = cc.getGroundTruthMasks(anns)
        for tup in segs:
            category_id, seg = tup
            maskSegmentOut(
                I, 
                seg, 
                running_mean_image,
                pic.replace(".jpg", "") + "_{}".format(category_id), 
                saveTo=out_folder
            )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ds", 
        "--dataset",
        default="train",
        help="which dataset to mask, of train, test, or val",
        choices=["train", "test", "val"]
    )
    args = parser.parse_args()
    mask_all(args)

if __name__ == '__main__':
    main()