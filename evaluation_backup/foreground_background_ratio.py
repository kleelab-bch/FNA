'''
Author Junbong Jang
Date 2/25/2020
'''
from __future__ import division
import cv2
import os
from statistics import mean 
dataset_folder = 'all-patients'

def convert_path_to_image(image_path):
    cur_image = cv2.imread(image_path)
    return cv2.cvtColor(cur_image, cv2.COLOR_BGR2GRAY)

def calc_ratio():
    mask_path = '../../DataSet_label/' + dataset_folder + '/mask/'
    ratio_list = []
    for mask_name in os.listdir(mask_path):
        if '.png' in mask_name or '.jpg' in mask_name:
            mask_image = convert_path_to_image(mask_path + mask_name)
            mask_white = (mask_image > 0).sum() # white regions
            mask_black = (mask_image == 0).sum() # white regions
            print(mask_white, mask_black, mask_white + mask_black, mask_white/mask_black)
            ratio_list.append(mask_white/mask_black)
    return ratio_list


if __name__ == "__main__":
    ratio_list = calc_ratio()
    print(len(ratio_list))
    print(mean(ratio_list))