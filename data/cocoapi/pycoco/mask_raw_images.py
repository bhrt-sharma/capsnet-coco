import os
import argparse
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from pycocotools.coco import COCO
from skimage.transform import resize
from scipy.ndimage import imread, gaussian_filter


TARGET_LENGTH = 640
TARGET_WIDTH = 640


def getIdFromImage(file_name):
    return int(file_name.split("_")[-1].replace(".jpg", ""))

def padImageToSquare(imArray, mode="edge"):
    length = imArray.shape[0]
    width = imArray.shape[1]

    length_diff = TARGET_LENGTH - length
    width_diff = TARGET_WIDTH - width 

    pad_width = ((length_diff // 2, length_diff // 2), (width_diff // 2, width_diff // 2), (0, 0))
    return np.pad(imArray, pad_width, mode)

def shrinkSquareImage(imArray, size=48):
    resized = resize(imArray, (size, size))
    rescaled_image = 255 * resized
    # Convert to integer data type pixels.
    final_image = rescaled_image.astype(np.uint8)
    return final_image

"""
@params:
- im is an **RGBA** Image as imported by scipy.ndimage.imread
- seg is segmentation array from the CoCoAPI
- out_name is what to call it 
- saveTo is a folder to save the image to 
- transparency determines whether the background is transparent (RGBA) or black (RGB)
"""
def maskSegmentOut(im, seg, background_image, out_name, saveTo="..", pad_to_square=False, shrink=False, normalize=False, grayscale=False):
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
    # newImArray[newImArray[:,:,3] == 0] = 0
    newImArray[newImArray[:,:,3] == 0] = np.concatenate([background_image])

    # mean center and unit variance by RGB channel
    if normalize:
        newImArray = (newImArray - np.mean(newImArray, axis=(0, 1))) / np.std(newImArray, axis=(0, 1))

    # pad to square and shrink
    if pad_to_square:
        newImArray = padImageToSquare(newImArray)
        if shrink:
            newImArray = shrinkSquareImage(newImArray)

    # gaussian filter 
    newImArray = gaussian_filter(newImArray[:, :, :3], 0.9)

    # back to Image from numpy
    if grayscale:
        mode = "L"
    else:
        mode = "RGB"

    newIm = Image.fromarray(newImArray[:, :, :3], mode)

    newIm.save("{}/{}.jpg".format(saveTo, out_name))

def mask_all(args):
    dataset = args.dataset
    pad_to_square = args.pad_to_square
    shrink = args.shrink
    black_and_white = args.black_and_white
    normalize = args.norm
    should_overwrite = args.overwrite

    # which folder to output to
    out_folder = "data/{}/images/masked".format(dataset)
    pics_already_masked = os.listdir(out_folder)
    pics_already_masked = [pic for pic in pics_already_masked if ".jpg" in pic]
    pics_already_masked = set(["_".join(pic.split("_")[:3]) + ".jpg" for pic in pics_already_masked])

    # initialize coco API
    ann_file = 'data/{}/instances_{}2014.json'.format(dataset, dataset)
    cc = COCO(ann_file)

    # loop through folder to get the mean image
    # have to do a running mean because there are too many images 
    # to store in memory at once 
    raw_folder = "data/{}/images/{}2014".format(dataset, dataset)
    all_pics = os.listdir(raw_folder)
    all_pics = [pic for pic in all_pics if ".jpg" in pic]

    # print("Computing per-channel means...")
    # running_mean_image = np.mean(imread("{}/{}".format(raw_folder, all_pics[0]), mode="RGB"), axis=(0,1))
    # num_images_processed = 1
    # for i in range(1, len(all_pics)):
    #     pic = all_pics[i]
    #     I = imread("{}/{}".format(raw_folder, pic), mode="RGB")
    #     img_mean = np.mean(I, axis=(0, 1))
    #     running_mean_image = running_mean_image * num_images_processed
    #     running_mean_image += img_mean
    #     num_images_processed += 1
    #     running_mean_image = running_mean_image / num_images_processed

    # then loop through to actually perform the masking 
    # this is done in RGBA and not RGB
    for pic in tqdm(all_pics):
        if not should_overwrite and pic in pics_already_masked:
            print("{} already masked.".format(pic))
            continue

        # get image and compute mean
        I = imread("{}/{}".format(raw_folder, pic), mode="RGBA")
        background_image = np.mean(I, axis=(0, 1))

        # converts something like "COCO_train2014_000000057870.jpg"
        # to 57870
        img_id = getIdFromImage(pic)

        # get annotation
        annIds = cc.getAnnIds(imgIds=img_id)
        anns = cc.loadAnns(annIds)

        # get segments from annotation
        segs = cc.getGroundTruthMasks(anns)

        category_counts = {} # allow for duplicates in each image file name
        for tup in segs:
            category_id, seg = tup

            # how many times have we seen this category id before in this image? 
            # be sure to put it in the f_name
            category_instance = category_counts.get(category_id, 1)
            category_counts[category_id] = category_instance + 1

            f_name = pic.replace(".jpg", "") + "_{}_{}".format(category_id, category_instance)
            maskSegmentOut(
                I, 
                seg, 
                background_image,
                f_name, 
                saveTo=out_folder,
                pad_to_square=pad_to_square,
                shrink=shrink,
                normalize=normalize,
                grayscale=black_and_white
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
        "-bw", 
        "--black_and_white",
        action='store_true',
        default=False,
        help="convert to grayscale"
    )
    parser.add_argument(
        "-ps", 
        "--pad_to_square",
        default=False,
        action='store_true',
        help="pad to square size"
    )
    parser.add_argument(
        "-sh", 
        "--shrink",
        default=False,
        action='store_true',
        help="shrink"
    )
    parser.add_argument(
        "-n", 
        "--norm",
        default=False,
        action='store_true',
        help="mean center and unit variance"
    )
    parser.add_argument(
        "-ow", 
        "--overwrite",
        default=False,
        action='store_true',
        help="if true, then overwrite files even if they exist already. if false, then check for existence"
    )

    args = parser.parse_args()
    
    if args.shrink and not args.pad_to_square:
        print("Can only shrink when padding to square.")
        return
    
    mask_all(args)

if __name__ == '__main__':
    main()
