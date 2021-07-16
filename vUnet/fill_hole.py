import cv2
import numpy as np
import os
from scipy import ndimage

'''
Author Junbong Jang
Date 10/7/2019

refer to https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
'''
def fill_hole_image(path, input_image):
    # Read image
    im_in = cv2.imread(path + '/' + input_image, cv2.IMREAD_GRAYSCALE)

    im_out_ndimage = ndimage.binary_fill_holes(im_in)


    # Threshold.
    # Set values equal to or above 220 to 0.
    # Set values below 220 to 255.
    th, im_th = cv2.threshold(im_in, 127, 255, cv2.THRESH_BINARY_INV)

    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv

    # Display images.
    # cv2.imshow("Thresholded Image", im_th)
    # cv2.imshow("Floodfilled Image", im_floodfill)
    # cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
    # cv2.imshow("Foreground", im_out)
    # cv2.waitKey(0)
    
    cv2.imwrite(path + '/filled/' + input_image, im_out_ndimage*255)
    #cv2.imwrite(path + '/filled2/' + input_image, im_out)

if __name__ == "__main__":
    dataset_folder = ['all-patients']
    #dataset_folder = ['gp2781-first', 'gp2781-fourth', 'gp2781-third', 'ha2779-first', 'ha2779-third']
    for folder in dataset_folder:
        predict_image_path = 'average_hist/predict_wholeframe/' + folder + '/' + folder
        for predicted_image_name in os.listdir(predict_image_path):
            if '.png' in predicted_image_name and 'filled-' not in predicted_image_name:
                print(predicted_image_name)
                fill_hole_image(predict_image_path, predicted_image_name)
           
