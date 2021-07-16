"""
Author Junbong Jang
Date: 8/18/2020

----rename stained images from each patient and combine them into one folder

split dataset into train and validation
convert RGB stained images into grayscale images
"""

import shutil
import cv2
import glob
import random
from PIL import Image


def RGB_to_gray_image(color_image_path, data_type):
    gray_image = cv2.imread(color_image_path, cv2.IMREAD_GRAYSCALE)
    im = Image.fromarray(gray_image)
    save_path = color_image_path.replace('/mask\\', f'/images_{data_type}/')
    im.save(save_path)


if __name__ == "__main__":
    random.seed(a=13)

    root_path = '//research.wpi.edu/leelab/Junbong/FNA/assets/all-patients-stained/'
    mask_list = glob.glob(root_path + 'mask/*.png')
    validation_sample_num = round(len(mask_list)*0.2)

    validation_mask_list = random.sample(mask_list, validation_sample_num)
    train_mask_list = list(set(mask_list) - set(validation_mask_list))

    print(len(mask_list))
    print(len(train_mask_list))
    print(len(validation_mask_list))
    print(train_mask_list)
    print(validation_mask_list)

    # ---------------------------------------------

    for train_mask_path in train_mask_list:
        print(train_mask_path)
        data_type = 'train'
        shutil.copy(train_mask_path, train_mask_path.replace('/mask\\', f'/masks_{data_type}/'))
        RGB_to_gray_image(train_mask_path, data_type)

    # for valid_mask_path in validation_mask_list:
    #     print(valid_mask_path)
    #     data_type = 'valid'
    #     shutil.copy(valid_mask_path, valid_mask_path.replace('/mask\\', f'/masks_{data_type}/'))
    #     RGB_to_gray_image(valid_mask_path, data_type)