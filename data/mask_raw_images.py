import argparse
from PIL import Image, ImageDraw
import numpy as np


"""
@params:
- im is an **RGBA** Image as imported by scipy.ndimage.imread
- seg is a segmentation array from the CoCoAPI
- fname is what to call it 
- saveTo is a folder to save the image to 
- transparency dcetermines whether the background is transparent (RGBA) or black (RGB)
"""
def maskSegmentOut(im, seg, fname, saveTo="..", transparency=False):
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
    newIm.save("{}/{}.jpg".format(saveTo, fname))
    
