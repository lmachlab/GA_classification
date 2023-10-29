import os
import numpy as np
import cv2
import random
import torch
from torch.utils.data import Dataset

def parse_image(img_name, img_folder, image_height, image_width):

    # Read the image with specified dimensions and 8-bit format
    img_path = os.path.join(img_folder, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    h,w,_ = img.shape
    if h != image_height or w != image_width:
        # Resize the image to the desired dimensions
        img = cv2.resize(img, (image_width, image_height))
    
    ## RANDOMLY APPLY AUGMENTATIONS HERE
    # flip_true = random.randint(0, 1)
    # if flip_true:
    #     img = cv2.flip(img, 1)
    
    return img


class DataGen(Dataset):
    def __init__(self, image_names, img_labels, img_folder, image_height=1536, image_width=500):
        self.image_height = image_height
        self.image_width = image_width
        self.image_names = image_names
        self.img_labels = img_labels
        self.img_folder = img_folder

    def __getitem__(self, index):
        
        image = parse_image(self.image_names[index], self.img_folder, self.image_height, self.image_width)
        label = self.img_labels[index]

        return image, label

    def __len__(self):
        return len(self.image_names)