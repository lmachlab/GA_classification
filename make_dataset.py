import os
import numpy as np
import cv2
import random
import torch
from torch.utils.data import Dataset

def parse_image(img_path, image_height, image_width):

    # Read the image with specified dimensions and 8-bit format
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    h,w,_ = img.shape
    if h != image_height or w != image_width:
        # Resize the image to the desired dimensions
        img = cv2.resize(img, (image_width, image_height))
    
    ## RANDOMLY APPLY AUGMENTATIONS HERE
    flip_true = random.randint(0, 1)
    if flip_true:
        img = cv2.flip(img, 1)

    
    return img


def find_label(img_path, img_labels):
    return


class DataGen(Dataset):
    def __init__(self, images_path, img_labels, image_height=1536, image_width=500):
        self.image_height = image_height
        self.image_width = image_width
        self.images_path = images_path
        self.img_labels = img_labels

    def __getitem__(self, index):
        
        image = parse_image(self.images_path[index], self.image_height, self.image_width)
        mask = find_label(self.images_path[index], self.img_labels)

        return image, mask

    def __len__(self):
        return len(self.images_path)