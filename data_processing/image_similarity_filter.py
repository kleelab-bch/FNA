"""
Author: Junbong Jang
Date: 6/30/2019

Cite: https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
import os


def operate_on_image_list():
    path = "../assets/unstained"

    patient_folders = os.listdir(path)
    patient_folders = [path + "/" + patient_folder for patient_folder in patient_folders]
    # print(patient_folders) ['../assets/stained/2775', '../assets/stained/2776', '../assets/stained/2777', '../assets/stained/2779', '../assets/stained/2780', '../assets/stained/2781']

    for patient_folder_path in patient_folders:  # iterate through the patient folders
        video_files = [f for f in os.listdir(patient_folder_path) if os.path.isfile(patient_folder_path + '/' + f)]
        print(f'patient folder: {patient_folder_path}')
        video_file_names = [video_file.replace('.mp4', '') for video_file in video_files]
        for index, video_file_name in enumerate(video_file_names): # iterate within a patient folder
            image_folder_path = patient_folder_path + '/' + video_file_name
            print('+++++++++++++++++++++ New Image Path List ++++++++++++++++++++')
            print(f'image folder: {image_folder_path}')
            image_path_list = [image_folder_path + '/' + f for f in os.listdir(image_folder_path) if
                               os.path.isfile(image_folder_path + '/' + f)]
            iterate_images(image_path_list)
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n\n\n')



def iterate_images(image_path_list):
    # loop over the input images
    blur_val_list = []
    group_image_path_list = []
    final_image_path_list = []
    for index, image_path in enumerate(image_path_list):
        # load the image, convert it to grayscale, and compute the
        # focus measure of the image using the Variance of Laplacian method
        cur_gray = convert_path_to_image(image_path)
        blurry_val = detect_blur(cur_gray)
        ssim_val = 'none'
        mse_val = 'none'
        if index > 0:  # compare between current to previous image to see if they are different.
            prev_image_path = image_path_list[index - 1]
            prev_gray = convert_path_to_image(prev_image_path)
            mse_val, ssim_val = compare_images(cur_gray, prev_gray)
            if ssim_val < 0.975:
                print('---------------group divided----------------')
                # for group_index, image_in_group in enumerate(group_image_path_list):
                #     print(blur_val_list[group_index], group_image_path_list[group_index])
                final_image_path = group_image_path_list[blur_val_list.index(max(blur_val_list))]
                print(group_image_path_list)
                print(blur_val_list)
                print(final_image_path)
                final_image_path_list.append(final_image_path)
                print('--------------------------------------------\n')
                blur_val_list = []
                group_image_path_list = []
        blur_val_list.append(blurry_val)
        group_image_path_list.append(image_path)

    delete_image_path_list = list(set(image_path_list) - set(final_image_path_list))
    for delete_image_path in delete_image_path_list:
        delete_file(delete_image_path)

        # show the image
        # cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        # cv2.imshow("Image", image)
        # print(image_path, blurry_val, ssim_val)


def convert_path_to_image(image_path):
    cur_image = cv2.imread(image_path)
    return cv2.cvtColor(cur_image, cv2.COLOR_BGR2GRAY)

def detect_blur(gray_image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(gray_image, cv2.CV_64F).var()

def mse(image_a, image_b):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((image_a.astype("float") - image_b.astype("float")) ** 2)
    err /= float(image_a.shape[0] * image_a.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compare_images(image_a, image_b):
    # compute the mean squared error and structural similarity
    # index for the images
    #mse_val = mse(image_a, image_b)
    ssim_val = ssim(image_a, image_b)

    return ssim_val

def findOnlyWhiteSpot(image_a, image_b):
    '''
    # first get locations of white spot from masked image
    # Then count the white spots in white locations of masked image
    '''

    if image_a.size != image_b.size or image_a.shape != image_b.shape:
        print('---------------error--------------------')
        return
    image_b_threshold = 127
    image_a_white_dots = (image_a == 255).sum()
    image_a_black_dots = (image_a == 0).sum()
    # print(image_a.shape)
    # print(image_a_white_dots)
    # print(image_a_black_dots)

    image_b_white_dots = (image_b >= image_b_threshold).sum()
    image_b_black_dots = (image_b < image_b_threshold).sum()

    # print(image_b.shape)
    # print(image_b_white_dots)
    # print(image_b_black_dots)

    if image_b_white_dots+image_b_black_dots != image_a_white_dots+image_a_black_dots:
        print('---------------error--------------------')
        return

    # for i in range(255):
    #     print((image_b == i).sum())

    white_overlap_inside = 0
    white_wrong_outside = 0
    for x in range(0, image_a.shape[0]):
        for y in range(0, image_a.shape[1]):
            if image_a[x, y] == 255 and image_b[x,y] > image_b_threshold:
                white_overlap_inside = white_overlap_inside + 1
            # if image_a[x, y] == 0 and image_b[x,y] > image_b_threshold:
            #     white_wrong_outside = white_wrong_outside + 1
    print('recall: {}'.format(white_overlap_inside/image_a_white_dots))
    print('precision: {}'.format(white_overlap_inside/image_b_white_dots))


if __name__ == "__main__":
    # detect_blur(["../assets/stained/2775/2775-1A-2-mag/image-0001.png"])

    # operate_on_image_list()

    # plt.imshow(cv2.imread('../assets/image/97-5 cb 100X.jpg'), cmap=plt.cm.gray)
    # plt.show()
    # first = convert_path_to_image('../assets/image/97-5 cb 100X.jpg')
    # plt.imshow(first, cmap=plt.cm.gray)
    # plt.show()
    # second = convert_path_to_image('../assets/stained/2775/2775-1A-2-mag/image-0049.png')
    # mse_val, ssim_val = compare_images(first, second)
    # print(ssim_val)

    first = convert_path_to_image('../../assets/DataSet_label/all-patients/mask/gp2781-first-0399.png')
    # plt.imshow(first, cmap=plt.cm.gray)
    # plt.show()

    second = convert_path_to_image('../../assets/DataSet_label/all-patients/mask/predictgp2781-first-0399.png')
    resized_second = cv2.resize(second, dsize=first.shape[::-1])
    # plt.imshow(resized_second, cmap=plt.cm.gray)
    # plt.show()
    findOnlyWhiteSpot(first, resized_second)
    # ssim_val = compare_images(first, resized_second)
    # print(ssim_val)

