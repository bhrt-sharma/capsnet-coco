# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 12:25:51 2017

@author: Yuxian Meng
"""

import argparse
import torch
from torchvision import datasets, transforms
from utils import load_tless_split

#TODO: data augmentation
#def augmentation(x, max_shift=2):
#    _, _, height, width = x.size()
#
#    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
#    source_height_slice = slice(max(0, h_shift), h_shift + height)
#    source_width_slice = slice(max(0, w_shift), w_shift + width)
#    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
#    target_width_slice = slice(max(0, -w_shift), -w_shift + width)
#
#    shifted_image = torch.zeros(*x.size())
#    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, 
#                 target_height_slice, target_width_slice]
#    return shifted_image.float()

def get_dataloader(cfg):
    # MNIST Dataset
    train_dataset, val, test_dataset = load_tless_split(cfg, num_classes=2)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=cfg.batch_size,
                                              shuffle=True)

    return train_loader, test_loader


if __name__ == "__main__":
    args = get_args()
    loader,_ = get_dataloader(args)
    print(len(loader.dataset))
    for data in loader:
        x,y = data
        print(x[0,0,:,:])
        break
