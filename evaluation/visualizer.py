'''
Author Junbong Jang
7/8/2020

Extracts bounding box coordinates from mask image which is inspired from Object Detection API
tensorflowAPI/object_detection/dataset_tools/create_faster_rcnn_tf_record_jj.py

Draws bounding box on the segmented grayscale image
Overlay bounding boxes from multiple sources onto one image
'''

import skimage.color
import skimage.filters
import skimage.io
# import skimage.viewer
import skimage.measure
import skimage.color
from skimage import img_as_ubyte
import cv2
import numpy as np
import os
from PIL import Image 
import PIL.ImageDraw as ImageDraw
from sklearn.linear_model import LinearRegression
import tensorflow as tf

from matplotlib import pyplot as plt
from scipy import ndimage
from count_overlap_box import *


class Visualizer:
    def __init__(self):
        print('Visualizer')


    def get_bounding_box_from_mask(self, mask, pixel_val):
        xmins = np.array([])
        ymins = np.array([])
        xmaxs = np.array([])
        ymaxs = np.array([])

        nonbackground_indices_x = np.any(mask == pixel_val, axis=0)
        nonbackground_indices_y = np.any(mask == pixel_val, axis=1)
        nonzero_x_indices = np.where(nonbackground_indices_x)
        nonzero_y_indices = np.where(nonbackground_indices_y)

        # if the mask contains any object
        if np.asarray(nonzero_x_indices).shape[1] > 0 and np.asarray(nonzero_y_indices).shape[1] > 0:

            mask_remapped = (mask == pixel_val).astype(np.uint8)  # get certain label
            mask_regions = skimage.measure.label(img_as_ubyte(mask_remapped), connectivity=2)  # get separate regions
            for region_pixel_val in np.unique(mask_regions):
                if region_pixel_val > 0: # ignore background pixels
                    # boolean array for localizing pixels from one region only
                    nonzero_x_boolean = np.any(mask_regions == region_pixel_val, axis=0)
                    nonzero_y_boolean = np.any(mask_regions == region_pixel_val, axis=1)

                    # numerical indices of value true in boolean array
                    nonzero_x_indices = np.where(nonzero_x_boolean)
                    nonzero_y_indices = np.where(nonzero_y_boolean)

                    # ignore small boxes
                    if len(nonzero_x_indices[0]) > 5 and len(nonzero_y_indices[0]) > 5:
                        xmin = float(np.min(nonzero_x_indices))
                        xmax = float(np.max(nonzero_x_indices))
                        ymin = float(np.min(nonzero_y_indices))
                        ymax = float(np.max(nonzero_y_indices))

                        print('bounding box for', region_pixel_val, xmin, xmax, ymin, ymax)

                        xmins = np.append(xmins, xmin)
                        ymins = np.append(ymins, ymin)
                        xmaxs = np.append(xmaxs, xmax)
                        ymaxs = np.append(ymaxs, ymax)

        # reshape 1xn row vector into nx1 column vector
        xmins = np.reshape(xmins, (-1, 1))
        ymins = np.reshape(ymins, (-1, 1))
        xmaxs = np.reshape(xmaxs, (-1, 1))
        ymaxs = np.reshape(ymaxs, (-1, 1))

        # bounding boxes in nx4 matrix
        bounding_boxes = np.concatenate((ymins, xmins, ymaxs, xmaxs), axis=1)
        return bounding_boxes


    def draw_bounding_boxes_on_image(self, image, boxes, color, thickness=8):
        """Draws bounding boxes on image.

        Args:
        boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
               The coordinates are in normalized format between [0, 1].
        color: color to draw bounding box. Default is red.
        thickness: line thickness. Default value is 4.

        Raises:
        ValueError: if boxes is not a [N, 4] array
        """
        boxes_shape = boxes.shape
        if not boxes_shape:
            return
        if len(boxes_shape) != 2 or boxes_shape[1] != 4:
            raise ValueError('Input must be of size [N, 4]')
        for i in range(boxes_shape[0]):
            self.draw_bounding_box_on_image(image, boxes[i, 0], boxes[i, 1], boxes[i, 2],
                                    boxes[i, 3], color, thickness)
        return image


    def draw_bounding_box_on_image(self, image, ymin, xmin, ymax, xmax, color, thickness=4):
        """Adds a bounding box to an image.

        Bounding box coordinates can be specified in either absolute (pixel) or
        normalized coordinates by setting the use_normalized_coordinates argument.

        Args:
        ymin: ymin of bounding box.
        xmin: xmin of bounding box.
        ymax: ymax of bounding box.
        xmax: xmax of bounding box.
        color: color to draw bounding box. Default is red.
        thickness: line thickness. Default value is 4.
        """
        draw = ImageDraw.Draw(image)
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
        draw.line([(left, top), (left, bottom), (right, bottom),
                    (right, top), (left, top)], width=thickness, fill=color)


    def vUNet_boxes(self, mask, a_threshold):
        # for vU-Net predictions
        # thresholding
        mask_copy = mask.copy()
        mask_copy[mask_copy >= a_threshold] = 255
        mask_copy[mask_copy < a_threshold] = 0
        mask_copy = self.clean_mask(mask_copy)
        mask_copy = self.clean_mask(mask_copy)

        # crop edge because of edge effect of vU-Net
        mask_copy = mask_copy[30:, 30:]
        mask_copy = np.pad(mask_copy, ((30,0), (30,0)), 'constant', constant_values=(0))

        return self.get_bounding_box_from_mask(mask_copy, pixel_val = 255)


    def clean_mask(self, input_mask):
        cleaned_mask = input_mask.copy()

        # fill hole
        cleaned_mask = ndimage.morphology.binary_fill_holes(cleaned_mask, structure=np.ones((5, 5))).astype(
            cleaned_mask.dtype)
        cleaned_mask = cleaned_mask * 255

        # Filter using contour area and remove small noise
        # https://stackoverflow.com/questions/60033274/how-to-remove-small-object-in-image-with-python
        cnts = cv2.findContours(cleaned_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, False)
            if area < 2000:
                contour_cmap = (0, 0, 0)
                cv2.drawContours(cleaned_mask, [c], -1, contour_cmap, -1)

        return cleaned_mask


    def open_mask(path):
        mask = Image.open(path)
        if mask.format != 'PNG':
            raise ValueError('Mask format is not PNG')
        mask = np.asarray(mask.convert('L'))
        return mask.copy()


    def mask_to_box(self, save_base_path, mask_root_path, mask_names, box_type):
        print('mask_tol_box')
        saved_boxes = {}
        for idx, filename in enumerate(mask_names):
            mask_path = os.path.join(mask_root_path, filename)
            mask_image = self.open_mask(mask_path)
            if 'ground_truth' in box_type:
                # saved_boxes[filename] = self.get_bounding_box_from_mask(mask_image, pixel_val = 76)
                saved_boxes[filename+'_76'] = self.get_bounding_box_from_mask(mask_image, pixel_val = 76)
                saved_boxes[filename+'_150'] = self.get_bounding_box_from_mask(mask_image, pixel_val = 150)
                saved_boxes[filename+'_29'] = self.get_bounding_box_from_mask(mask_image, pixel_val = 29)
            elif box_type == 'vunet':
                a_threshold = 100
                saved_boxes[filename] = self.vUNet_boxes(mask_image, a_threshold)
            else:
                print('box type None')

        np.save(save_base_path + '{}_boxes.npy'.format(box_type), saved_boxes)


    def overlay_boxes_ensemble(self, save_base_path, img_root_path, mask_names, ground_truth_boxes, unstained_detection_boxes, stained_detection_boxes, predict_box_type):
        '''
            get bounding box from vUNet and Faster Rcnn models, and ground truth mask.
            overlay them all on top of the unstained raw image
        '''

        total_false_negative = 0
        total_false_positive = 0
        total_gt_overlaps = 0

        for idx, filename in enumerate(mask_names):

            img_path = os.path.join(img_root_path, filename.replace('predict', ''))
            img = Image.open(img_path)  # load images from paths

            one_ground_truth_boxes = ground_truth_boxes.item()[filename]
            one_unstained_predicted_boxes = unstained_detection_boxes.item()[filename]
            one_stained_predicted_boxes = stained_detection_boxes.item()[filename]

            if predict_box_type == 'faster_rcnn':
                one_unstained_predicted_boxes[:, 0], one_unstained_predicted_boxes[:, 2] = one_unstained_predicted_boxes[:,0] * img.height, one_unstained_predicted_boxes[:, 2] * img.height  # 1944
                one_unstained_predicted_boxes[:, 1], one_unstained_predicted_boxes[:, 3] = one_unstained_predicted_boxes[:,1] * img.width, one_unstained_predicted_boxes[:, 3] * img.width  # 2592
                one_stained_predicted_boxes[:, 0], one_stained_predicted_boxes[:, 2] = one_stained_predicted_boxes[:,0] * img.height, one_stained_predicted_boxes[:, 2] * img.height  # 1944
                one_stained_predicted_boxes[:, 1], one_stained_predicted_boxes[:, 3] = one_stained_predicted_boxes[:,1] * img.width, one_stained_predicted_boxes[:, 3] * img.width  # 2592

            one_predicted_boxes = np.concatenate((one_unstained_predicted_boxes, one_stained_predicted_boxes), axis=0)
            # Save image with bounding box
            save_path = os.path.join(save_base_path, filename.replace('predict', ''))
            print(save_path)

            if one_ground_truth_boxes.shape[0] > 0 or one_predicted_boxes.shape[0] > 0:
                combined_boxed_image = self.draw_bounding_boxes_on_image(img, one_ground_truth_boxes, color='#9901ff')
                combined_boxed_image = self.draw_bounding_boxes_on_image(combined_boxed_image, one_predicted_boxes,
                                                                    color='#89ff29')
                combined_boxed_image.save(save_path)

                gt_overlaps, false_negative, false_positive = count_overlap_box(one_ground_truth_boxes, one_predicted_boxes)
                print(gt_overlaps, false_negative, false_positive)
                total_gt_overlaps = total_gt_overlaps + gt_overlaps
                total_false_negative = total_false_negative + false_negative
                total_false_positive = total_false_positive + false_positive
            else:
                print()

        print('tp:', total_gt_overlaps, 'fn:', total_false_negative, 'fp:', total_false_positive)
        precision = total_gt_overlaps / (total_gt_overlaps + total_false_positive)
        recall = total_gt_overlaps / (total_gt_overlaps + total_false_negative)
        f1 = 2*precision*recall/(precision+recall)
        print('precision:', precision)
        print('recall:', recall)
        print('f1:', f1)


    def overlay_boxes(self, save_base_path, img_root_path, mask_names, ground_truth_boxes, prediction_boxes, predict_box_type):
        '''
            get bounding box from vUNet and Faster Rcnn models, and ground truth mask.
            overlay them all on top of the unstained raw image
        '''
        total_false_negative = 0
        total_false_positive = 0
        total_gt_overlaps = 0

        for idx, filename in enumerate(mask_names):

            img_path = os.path.join(img_root_path, filename)
            img = Image.open(img_path)  # load images from paths

            orig_filename = filename.replace('predict', '').replace('stained_','')
            one_ground_truth_boxes = ground_truth_boxes.item()[orig_filename]
            one_predicted_boxes = prediction_boxes.item()[filename]

            if predict_box_type == 'faster_rcnn':
                one_predicted_boxes[:, 0], one_predicted_boxes[:, 2] = one_predicted_boxes[:,0] * img.height, one_predicted_boxes[:, 2] * img.height  # 1944
                one_predicted_boxes[:, 1], one_predicted_boxes[:, 3] = one_predicted_boxes[:,1] * img.width, one_predicted_boxes[:, 3] * img.width  # 2592

            # Save image with bounding box
            save_path = os.path.join(save_base_path, filename.replace('predict', ''))
            print(save_path)

            if one_ground_truth_boxes.shape[0] > 0 or one_predicted_boxes.shape[0] > 0:
                # Draw only ground truth boxes for patent figures
                # blank_image = Image.new(mode='RGB', size=(img.width, img.height), color=0)
                # one_ground_truth_boxes = ground_truth_boxes.item()[filename+'_29']
                # two_ground_truth_boxes = ground_truth_boxes.item()[filename+'_76']
                # three_ground_truth_boxes = ground_truth_boxes.item()[filename+'_150']
                # combined_boxed_image = self.draw_bounding_boxes_on_image(blank_image, one_ground_truth_boxes, color='blue', thickness=16)
                # combined_boxed_image = self.draw_bounding_boxes_on_image(combined_boxed_image, two_ground_truth_boxes, color='red', thickness=16)
                # combined_boxed_image = self.draw_bounding_boxes_on_image(combined_boxed_image, three_ground_truth_boxes, color='green', thickness=16)

                combined_boxed_image = self.draw_bounding_boxes_on_image(img, one_ground_truth_boxes, color='#9901ff')
                combined_boxed_image = self.draw_bounding_boxes_on_image(combined_boxed_image, one_predicted_boxes,
                                                                    color='#89ff29')
                combined_boxed_image.save(save_path)

                gt_overlaps, false_negative, false_positive = count_overlap_box(one_ground_truth_boxes, one_predicted_boxes)
                print(gt_overlaps, false_negative, false_positive)
                total_gt_overlaps = total_gt_overlaps + gt_overlaps
                total_false_negative = total_false_negative + false_negative
                total_false_positive = total_false_positive + false_positive
            else:
                print()

        print('tp:', total_gt_overlaps, 'fn:', total_false_negative, 'fp:', total_false_positive)
        precision = total_gt_overlaps / (total_gt_overlaps + total_false_positive)
        recall = total_gt_overlaps / (total_gt_overlaps + total_false_negative)
        f1 = 2*precision*recall/(precision+recall)
        print('precision:', precision)
        print('recall:', recall)
        print('f1:', f1)


    def bounding_box_per_image_distribution(self, save_base_path, mask_names, ground_truth_boxes, predicted_boxes, predict_box_type):
        '''
        scatter plot of ground truth box vs. predicted box

        :param mask_names:
        :param ground_truth_boxes:
        :param predicted_boxes:
        :return:
        '''

        MAX_BOX_NUM = 18
        ground_truth_box_nums = []
        predicted_box_nums = []
        for idx, filename in enumerate(mask_names):
            one_ground_truth_boxes = ground_truth_boxes.item()[filename]
            one_predicted_boxes = predicted_boxes.item()[filename]
            ground_truth_box_nums.append(one_ground_truth_boxes.shape[0])
            predicted_box_nums.append(one_predicted_boxes.shape[0])

        print(max(ground_truth_box_nums), max(predicted_box_nums))

        fig = plt.figure()  # pylab.figure()
        ax = fig.add_subplot(111)

        plt.scatter(ground_truth_box_nums, predicted_box_nums, alpha=1)

        # linear regression of scatter plot
        reg = LinearRegression().fit(np.asarray(ground_truth_box_nums).reshape(-1, 1), predicted_box_nums)
        r_squared = reg.score(np.asarray(ground_truth_box_nums).reshape(-1, 1), predicted_box_nums)
        m = reg.coef_[0]
        b = reg.intercept_

        x_range = np.arange(MAX_BOX_NUM+1)
        plt.plot(x_range, x_range, color='red')
        plt.plot(x_range, m * x_range + b)
        ax.text(0.07, 0.89, f'y=x*{round(m, 3)}+{round(b, 3)}\n$R^2$={round(r_squared, 3)}', color='#1f77b4',
                horizontalalignment='left',
                verticalalignment='center',
                transform=ax.transAxes)


        plt.title(f'Ground Truth Vs. {predict_box_type} Per Image', fontsize='x-large')
        plt.xlabel('Ground Truth Number of Follicular Clusters', fontsize='large')
        plt.ylabel('Predicted Number of Follicular Clusters', fontsize='large')

        plt.xlim(left=-1, right=MAX_BOX_NUM)
        plt.ylim(bottom=-1, top=MAX_BOX_NUM)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.savefig(save_base_path + 'image_line_graph.png')


    def visualize_feature_activation_map(self, feature_map, image_path, image_name, save_path):
        # reference https://codeocean.com/capsule/0685076/tree/v1
        # reduce feature map depth from 512 to 1
        averaged_feature_map = np.zeros(feature_map.shape[:2], dtype=np.float64)
        for i in range(feature_map.shape[0]):
            for j in range(feature_map.shape[1]):
                for k in range(feature_map.shape[2]):
                    averaged_feature_map[i,j] = averaged_feature_map[i,j] + feature_map[i,j,k]
        # get image to overlay
        image = cv2.imread(image_path + image_name)
        width, height, channels = image.shape

        # generate heatmap
        cam = averaged_feature_map / np.max(averaged_feature_map)
        cam = cv2.resize(cam, (height, width))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

        # overlay heatmap on original image
        alpha = 0.6
        heatmap_img = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        cv2.imwrite(f'{save_path}heatmap_{image_name}', heatmap_img)


