#%% md
'''
Author: Junbong Jang

Date: 11/25/2021
'''

import numpy as np
import skimage
import glob
import cv2
from skimage import img_as_ubyte
import skimage.measure


def get_separate_regions(mask):
    return skimage.measure.label(img_as_ubyte(mask), connectivity=2)


def convert_box_coordinates_to_center_boxes(box_coordinate, img_width, img_height):
    # input box_coordinate: ymin, xmin, ymax, xmax
    # output box_coord: x_center_coord, y_center_coord, width, height

    box_coordinate = np.array(box_coordinate)
    new_box_coord = np.zeros(box_coordinate.shape)
    # print('input', box_coordinate)

    new_box_coord[:,0] = (box_coordinate[:,0] + box_coordinate[:,2]) / 2 # y center coord
    new_box_coord[:,1] = (box_coordinate[:,1] + box_coordinate[:,3]) / 2 # x center coord
    new_box_coord[:,2] = box_coordinate[:,3] - box_coordinate[:,1] # width
    new_box_coord[:,3] = box_coordinate[:,2] - box_coordinate[:,0] # height

    inter_box_coord = np.zeros(box_coordinate.shape)
    inter_box_coord[:,0] = new_box_coord[:, 1]  # x center coord
    inter_box_coord[:,1] = new_box_coord[:, 0]  # y center coord
    inter_box_coord[:,2] = new_box_coord[:, 2]  # width
    inter_box_coord[:,3] = new_box_coord[:, 3]  # height

    inter_box_coord[:,0], inter_box_coord[:,2] = inter_box_coord[:,0] / img_width, inter_box_coord[:,2] / img_width
    inter_box_coord[:,1], inter_box_coord[:,3] = inter_box_coord[:,1] / img_height, inter_box_coord[:,3] / img_height

    inter_box_coord = np.around(inter_box_coord, decimals=6)

    inter_box_coord = inter_box_coord.tolist()

    return inter_box_coord


def get_mask(path):

    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def save_bounding_boxes(save_dir, save_filename, class_id_list, bounding_boxes):
    with open(save_dir + save_filename, 'w') as f:
        for class_id, a_bounding_box in zip(class_id_list, bounding_boxes):
            str1 = ' '.join(str(e) for e in a_bounding_box)
            str1 = f'{class_id} ' + str1 + '\n'
            f.write(str1)


def class_text_to_pixel_val(class_name):
    if class_name == 'follicular':
        return 76
    elif class_name == 'secretion':
        return 149
    elif class_name == 'artifact':
        return 29
    else:
        None


def get_image_filename(image_filepath, data_type):
    image_filename = image_filepath.split("/")[-1].replace(f'masks_{data_type}\\', '')[:-4]
    return image_filename


if __name__ == "__main__":
    for data_type in ['train', 'validation', 'test']:
        mask_path = f'../assets/masks_{data_type}/'
        save_label_filepath = f'../assets/{data_type}/labels/'
        img_format = '.png'
        label_format = '.txt'

        mask_list = glob.glob(mask_path + '*' + img_format)

        for mask_path in mask_list:
            mask = get_mask(mask_path)
            print(np.unique(mask))
            img_height, img_width = mask.shape
            mask_name = get_image_filename(mask_path, data_type)

            # get bounding box coordinates from mask
            box_coordinates = []
            class_id_list = []

            no_detection_flag = True
            for class_name in ['follicular', 'secretion', 'artifact']:  # label_map_dict.keys():
                pixel_val = class_text_to_pixel_val(class_name)
                nonbackground_indices_x = np.any(mask == pixel_val, axis=0)
                nonbackground_indices_y = np.any(mask == pixel_val, axis=1)
                nonzero_x_indices = np.where(nonbackground_indices_x)
                nonzero_y_indices = np.where(nonbackground_indices_y)

                if np.asarray(nonzero_x_indices).shape[1] > 0 and np.asarray(nonzero_y_indices).shape[1] > 0:
                    # if the mask contains any object
                    #   do regionprops to get individual regions

                    mask_remapped = (mask == pixel_val).astype(np.uint8)
                    mask_regions = get_separate_regions(mask_remapped)
                    for region_pixel_val in np.unique(mask_regions):
                        # draw bounding box over each region labeled in mask
                        if region_pixel_val > 0:  # not background pixels

                            # boolean array
                            nonzero_x_boolean = np.any(mask_regions == region_pixel_val, axis=0)
                            nonzero_y_boolean = np.any(mask_regions == region_pixel_val, axis=1)

                            # numerical indices only for true in boolean array
                            nonzero_x_indices = np.where(nonzero_x_boolean)
                            nonzero_y_indices = np.where(nonzero_y_boolean)

                            # remove small size boxes
                            print('bounding box size: ', len(nonzero_x_indices[0]), len(nonzero_y_indices[0]))
                            if len(nonzero_x_indices[0]) > 5 and len(nonzero_y_indices[0]) > 5:
                                no_detection_flag = False
                                xmin = float(np.min(nonzero_x_indices))
                                xmax = float(np.max(nonzero_x_indices))
                                ymin = float(np.min(nonzero_y_indices))
                                ymax = float(np.max(nonzero_y_indices))

                                class_id = 0
                                if class_name != 'follicular':  # added to combine two negative control labels together in 6/30/2020
                                    class_name = 'negative'
                                    class_id = 1
                                class_id_list.append(class_id)
                                print('bounding box info: ', class_name, region_pixel_val, xmin, xmax, ymin, ymax)

                                box_coordinates.append([ymin, xmin, ymax, xmax])

            if len(box_coordinates) != 0:
                center_box_coordinates = convert_box_coordinates_to_center_boxes(box_coordinates, img_width, img_height)
                save_bounding_boxes(save_label_filepath, mask_name + label_format, class_id_list, center_box_coordinates)
                print('-----------------')

