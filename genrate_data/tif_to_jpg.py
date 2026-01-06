import re
from tqdm import tqdm
import os
import argparse
from os.path import join
import cv2
VIS_file_path = '/mnt/workspace/datasets/CVC-14/cvc14_train_visible.txt'


file = open(VIS_file_path, 'r')
img_paths = file.read().splitlines()

os.makedirs('training_cvc14_images', exist_ok=True)
index  = range(0, len(img_paths))

path_dict = dict(zip(img_paths, index))

for img_path in img_paths:
    img = cv2.imread(img_path)
    cv2.imwrite(f'training_cvc14_images/{path_dict[img_path]}_rgb_train.jpg',img)
