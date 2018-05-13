import argparse
from PIL import Image, ImageDraw
import numpy as np
import os
from pycocotools.coco import COCO
from scipy.ndimage import imread


"""
@params:
- im is an **RGBA** Image as imported by scipy.ndimage.imread
- seg is segmentation array from the CoCoAPI
- out_name is what to call it 
- saveTo is a folder to save the image to 
- transparency determines whether the background is transparent (RGBA) or black (RGB)
"""
def maskSegmentOut(im, seg, out_name, saveTo="..", transparency=False):
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
    if not transparency:
        newImArray[newImArray[:,:,3] == 0] = 0
        # back to Image from numpy
        newIm = Image.fromarray(newImArray[:, :, :3], "RGB")
    else:
        newIm = Image.fromarray(newImArray, "RGBA")
    newIm.save("{}/{}.jpg".format(saveTo, out_name))

def mask_all(args):
    # decide which folder to output to
    remove_texture = args.remove_texture
    dataset = args.dataset
    out_folder = "data/{}/images/".format(dataset)
    if remove_texture:
        out_folder += "masked_low_contrast"
    else:
        out_folder += "masked"

    # initialize coco API
    ann_file = 'data/{}/instances_{}2014.json'.format(dataset, dataset)
    cc = COCO(ann_file)

    # loop through folder
    raw_folder = "data/{}/images/{}2014".format(dataset, dataset)
    all_pics = os.listdir(raw_folder)
    for pic in all_pics:
        if ".jpg" in pic:
            # get image 
            I = imread("{}/{}".format(raw_folder, pic), mode="RGBA")

            # converts something like "COCO_train2014_000000057870.jpg"
            # to 57870
            img_id = int(pic.split("_")[-1].replace(".jpg", ""))

            # get annotation
            annIds = cc.getAnnIds(imgIds=img_id)
            anns = cc.loadAnns(annIds)

            # get segments from annotation
            segs = cc.getGroundTruthMasks(anns)
            for tup in segs:
                category_id, seg = tup
                maskSegmentOut(
                    I, 
                    tup, 
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
    parser.add_argument(
        "-rt", 
        "--remove_texture",
        default=False,
        help="whether or not to remove texture",
    )
    args = parser.parse_args()
    mask_all(args)

if __name__ == '__main__':
    main()