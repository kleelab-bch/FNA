from __future__ import division
import os
import cv2
import matplotlib.pyplot as plt
import collections
import pandas as pd
from skimage.measure import compare_ssim as ssim
from keras import backend as K
import tensorflow as tf
sess = tf.InteractiveSession()
'''
Author Junbong Jang
Date: 9/30/2019

learn about precision and recall here 
https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9
'''

def convert_path_to_image(image_path):
    cur_image = cv2.imread(image_path)
    return cv2.cvtColor(cur_image, cv2.COLOR_BGR2GRAY)

def findOnlyWhiteSpot(image_a, image_b):
    '''
    # first get locations of white spot from masked image
    # Then count the white spots in white locations of masked image
    '''

    if image_a.size != image_b.size or image_a.shape != image_b.shape:
        print('---------------image size error--------------------')
        return
    white_threshold = 127
    image_a_white_dots = (image_a >= white_threshold).sum() # actual positive
    image_a_black_dots = (image_a < white_threshold).sum() # actual positive
    # print(image_a.shape)
    # print(image_a_white_dots)
    # print(image_a_black_dots)

    image_b_white_dots = (image_b >= white_threshold).sum() # predicted positive
    image_b_black_dots = (image_b < white_threshold).sum() # predicted negative

    # print(image_b.shape)
    # print(image_b_white_dots)
    # print(image_b_black_dots)

    if image_b_white_dots+image_b_black_dots != image_a_white_dots+image_a_black_dots:
        print('---------------dots error--------------------')
        print(image_b_white_dots, image_b_black_dots)
        print(image_a_white_dots, image_a_black_dots)
        return

    white_overlap_inside = 0  # true positive
    black_outside = 0  # true negative
    for x in range(0, image_a.shape[0]):
        for y in range(0, image_a.shape[1]):
            if image_a[x, y] >= white_threshold and image_b[x,y] >= white_threshold:
                white_overlap_inside = white_overlap_inside + 1
            #if image_a[x, y] < white_threshold and image_b[x,y] < white_threshold:
            #    black_outside = black_outside + 1
    
    false_positive = image_b_white_dots - white_overlap_inside
    false_negative = image_a_white_dots - white_overlap_inside
    
    recall = white_overlap_inside/(white_overlap_inside + false_negative)
    precision = white_overlap_inside/(white_overlap_inside + false_positive)
    harmony = 2*recall*precision/(recall+precision)
    ssim_val = ssim(image_a, image_b)
    dice_val = dice_coef(image_a, image_b)
    dice_val = dice_val.eval()
    
    #print(white_overlap_inside)
    #print(image_a_white_dots)
    #print(image_b_white_dots)
    print('recall: {}'.format(recall))
    print('precision: {}'.format(precision))
    print('harmony: {}'.format(harmony))
    print('ssim_val: {}'.format(ssim_val))
    print('dice_val: {}'.format(dice_val))
    
    return recall, precision, harmony, ssim_val, dice_val

def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(K.round(y_true))
    y_pred_f = K.flatten(K.round(y_pred))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

if __name__ == "__main__":
    plt.switch_backend('agg') # needed for plt.savefig('hello.png')
    #dataset_folder = ['bryan']
    dataset_folder = ['all-patients']
    #dataset_folder = ['gp2781-first', 'gp2781-fourth', 'gp2781-third', 'ha2779-first', 'ha2779-third']
    for folder in dataset_folder:
        result_dict = collections.defaultdict(dict)
        predict_image_path = '../Step2_vUnet/average_hist/predict_wholeframe/' + folder + '/' + folder

        for predicted_image_name in os.listdir(predict_image_path):
            print(predicted_image_name)
            image_name = predicted_image_name.replace('predict','')
            if '.png' in image_name or '.jpg' in image_name:
                mask_image = convert_path_to_image('../../DataSet_label/' + folder + '/mask/' + image_name)
                
                predicted_image = convert_path_to_image(predict_image_path + '/' + predicted_image_name)
                resized_predicted_image = cv2.resize(predicted_image, dsize=mask_image.shape[::-1])
                result_dict[image_name] = (findOnlyWhiteSpot(mask_image, resized_predicted_image))
        df = pd.DataFrame.from_dict(result_dict)
        df = df.swapaxes("index", "columns")
        df = df.sort_index()
        df = df.rename(columns={0: "recall", 1: "precision", 2: "harmony", 3: "ssim_val", 4: 'dice_coef'})
        df.to_excel('prediction_result_'+ folder +'.xlsx')
