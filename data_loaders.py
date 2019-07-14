import os
import cv2
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt


def get_single_image_patches(image_path, n=100, patch_size=160, resize=None):
    image = cv2.imread(image_path)
    patches = []
    h, w = image.shape[:2]

    for _ in range(n):
        random_h = np.random.randint(0, h - patch_size)
        random_w = np.random.randint(0, w - patch_size)
        patch = image[random_h:random_h+patch_size, random_w:random_w+patch_size, :]
        if resize:
            patch = cv2.resize(patch, (resize, resize))
        patches.append(patch)
    
    patches = np.array([np.array(x / 255.0) for x in patches])
    patches = np.transpose(patches, (0, 3, 1, 2))
    patches = torch.Tensor(patches)
    
    return patches


def load_dtd(dtd_dir, n=120, patch_size=300, patches_per_image=1, resize=None):
    image_paths = os.listdir(dtd_dir)
    image_paths = [dtd_dir + x for x in image_paths]

    images = []

    for path in image_paths:
        image = cv2.imread(path)
        h, w = image.shape[:2]
        for _ in range(patches_per_image):
            random_h = np.random.randint(0, h - patch_size)
            random_w = np.random.randint(0, w - patch_size)
            patch = image[random_h:random_h+patch_size, random_w:random_w+patch_size, :]
            if resize:
                patch = cv2.resize(patch, (resize, resize))
            images.append(patch)
    
    images = np.array([np.array(x / 255.0) for x in images])
    images = np.transpose(images, (0, 3, 1, 2))
    images = torch.Tensor(images)
    
    return images
