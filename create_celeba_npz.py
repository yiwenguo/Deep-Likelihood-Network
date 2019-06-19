#!/usr/bin/env python
import math
import numpy as np

from PIL import Image
from random import sample
from os import listdir, mkdir
from os.path import exists, isfile, join


# load image path
def load_image_path(dataset_root, subset):
    file_path = join(dataset_root, subset, 'images')
    onlyfiles = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    return onlyfiles


def train_val_test_npz():
    dataset_root = './data/CelebA'   #/path/to/your/CelebA/
    if 'numpy' not in listdir(dataset_root):
        mkdir(dataset_root+'/numpy')  
    subsets = ['train', 'val', 'test']
    image_size = [64, 64]
    for subset in subsets:
        print ('Finding %sing image files...' %subset)
        idx_base, image_array_resized = 0, []
        onlyfiles = load_image_path(dataset_root, subset)
        if subset=='train':
            onlyfiles = sample(onlyfiles, len(onlyfiles))
        for idx, image in enumerate(onlyfiles):
            fname = join(dataset_root, subset, 'images', image)
            image = Image.open(fname)
            image_height, image_width = np.shape(image)[:2]
            image_resized = image.resize(image_size, Image.BICUBIC)
            image_array_resized.append(np.array(image_resized))  
            n = idx - idx_base + 1
            if n % 102400 == 0:
                np.savez(dataset_root + '/numpy/%s_image_array_%s.npz' 
                         %(subset, int(math.floor(n / 102400) - 1)), 
                         images=image_array_resized)
                print ('\t%d / %d finished' % (n, len(onlyfiles)))
                image_array_resized = []
        np.savez(dataset_root + '/numpy/%s_image_array_%s.npz' 
                 %(subset, int(math.floor(n / 102400))), 
                 images=image_array_resized)
        print ('\t%d / %d finished' % (n, len(onlyfiles)))     


if __name__ == '__main__':
    train_val_test_npz()

