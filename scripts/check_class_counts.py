import os
import collections


def check(folder='data/train/images/final'):
    imgs = os.listdir(folder)
    imgs = [f for f in imgs if ".jpg" in f]
    categories = [f.split("_")[-2] for f in imgs]
    counter = collections.Counter(categories)
    print(folder)
    print(counter)

check('data/train/images/final')
check('data/test/images/final')
check('data/val/images/final')
    
