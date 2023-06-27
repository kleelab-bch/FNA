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
import skimage.measure
import skimage.color
from skimage import img_as_ubyte
import cv2
import os
from PIL import Image 
import PIL.ImageDraw as ImageDraw
from sklearn.linear_model import LinearRegression
import numpy as np

import matplotlib
from matplotlib import pyplot as plt
from scipy import ndimage
from calc_polygon import *
from sklearn.metrics import auc

import rasterio
from rasterio import features
import shapely
from shapely.geometry import Point, Polygon
from shapely.geometry.multipolygon import MultiPolygon

import seaborn as sns
import pandas as pd

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

font = {'size':'11', 'weight':'normal',}
matplotlib.rc('font', **font)


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


    def draw_bounding_boxes(self, image, boxes, color, thickness=8):
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
            self.draw_bounding_box(image, boxes[i, 0], boxes[i, 1], boxes[i, 2],
                                    boxes[i, 3], color, thickness)
        return image


    def draw_bounding_box(self, image, ymin, xmin, ymax, xmax, color, thickness=4):
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


    def open_mask(self, path):
        mask = Image.open(path)
        if mask.format != 'PNG':
            raise ValueError('Mask format is not PNG')
        mask = np.asarray(mask.convert('L'))
        return mask.copy()


    def mask_to_box(self, save_base_path, mask_root_path, mask_names, box_type):
        print('mask_to_box')
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


    def mask_to_polygons(self, mask_root_path, mask_names):
        list_of_polygons = []
        for idx, filename in enumerate(mask_names):
            mask_path = os.path.join(mask_root_path, filename)
            a_mask = self.open_mask(mask_path)
            polygons = self.mask_to_polygons_layer(a_mask)
            list_of_polygons.append(polygons)

        return list_of_polygons


    def mask_to_polygons_layer(self, mask):
        # refer to https://rocreguant.com/convert-a-mask-into-a-polygon-for-images-using-shapely-and-rasterio/1786/
        all_polygons = []
        for shape, value in features.shapes(mask.astype(np.uint8), mask=(mask > 0), transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):
            all_polygons.append(shapely.geometry.shape(shape))

        all_polygons = gpd.GeoSeries(unary_union(all_polygons))[0]
        #
        # all_polygons = shapely.geometry.MultiPolygon(all_polygons)
        # if not all_polygons.is_valid:
        #     all_polygons = all_polygons.buffer(0)
        #     # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        #     # need to keep it a Multi throughout
        #     if all_polygons.type == 'Polygon':
        #         all_polygons = shapely.geometry.MultiPolygon([all_polygons])
        return all_polygons


    def overlay_boxes_over_images(self, save_base_path, img_root_path, mask_names, ground_truth_boxes, prediction_boxes, model_type):
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

            if 'faster' in model_type:
                orig_filename = filename.replace('predict', '').replace('stained_', '')
                one_ground_truth_boxes = ground_truth_boxes.item()[orig_filename]
                one_predicted_boxes = prediction_boxes.item()[filename]

                one_predicted_boxes[:, 0], one_predicted_boxes[:, 2] = one_predicted_boxes[:,0] * img.height, one_predicted_boxes[:, 2] * img.height  # 1944
                one_predicted_boxes[:, 1], one_predicted_boxes[:, 3] = one_predicted_boxes[:,1] * img.width, one_predicted_boxes[:, 3] * img.width  # 2592

            else:
                # 7/27/2021 for FNA MTL classifier
                one_ground_truth_boxes = ground_truth_boxes.item()[idx]
                one_predicted_boxes = prediction_boxes.item()[idx]
                print(idx, filename)
                print(one_ground_truth_boxes.shape, one_predicted_boxes.shape)

            # Save image with bounding box
            save_path = os.path.join(save_base_path, filename.replace('predict', ''))

            if one_ground_truth_boxes.shape[0] > 0 or one_predicted_boxes.shape[0] > 0:
                # Draw only ground truth boxes for patent figures
                # blank_image = Image.new(mode='RGB', size=(img.width, img.height), color=0)
                # one_ground_truth_boxes = ground_truth_boxes.item()[filename+'_29']
                # two_ground_truth_boxes = ground_truth_boxes.item()[filename+'_76']
                # three_ground_truth_boxes = ground_truth_boxes.item()[filename+'_150']
                # combined_boxed_image = self.draw_bounding_boxes(blank_image, one_ground_truth_boxes, color='blue', thickness=16)
                # combined_boxed_image = self.draw_bounding_boxes(combined_boxed_image, two_ground_truth_boxes, color='red', thickness=16)
                # combined_boxed_image = self.draw_bounding_boxes(combined_boxed_image, three_ground_truth_boxes, color='green', thickness=16)

                combined_boxed_image = self.draw_bounding_boxes(img, one_ground_truth_boxes, color=(255,0,0))
                combined_boxed_image = self.draw_bounding_boxes(combined_boxed_image, one_predicted_boxes, color=(0, 255, 0))

                gt_overlaps, false_negative, false_positive, gt_overlap_pairs = count_overlap_box(one_ground_truth_boxes, one_predicted_boxes)
                print(filename)
                print(gt_overlaps, false_negative, false_positive)

                # since PIL draw do not blend lines, I manually draw the overlapped boxes with different color
                overlapped_ground_truth_box_indices = [gt_overlap_pair[0] for gt_overlap_pair in gt_overlap_pairs]
                combined_boxed_image = self.draw_bounding_boxes(combined_boxed_image, one_ground_truth_boxes[overlapped_ground_truth_box_indices],
                                                                         color=(255, 255, 0))

                combined_boxed_image.save(save_path)

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


    def overlay_polygons_over_images(self, save_base_path, img_root_path, mask_names, ground_truth_boxes, prediction_boxes, model_type):
        print('overlay_polygons_over_images')
        pred_area_of_all_images = 0
        gt_area_of_all_images = 0
        intersect_area_of_all_images = 0
        union_area_of_all_images = 0

        for idx, filename in enumerate(mask_names):
            img_path = os.path.join(img_root_path, filename)
            img = Image.open(img_path)  # load images from paths
            print(idx, filename)

            if 'faster' in model_type:
                orig_filename = filename.replace('predict', '').replace('stained_', '')
                one_ground_truth_boxes = ground_truth_boxes.item()[orig_filename]
                one_predicted_boxes = prediction_boxes.item()[filename]

                one_predicted_boxes[:, 0], one_predicted_boxes[:, 2] = one_predicted_boxes[:,0] * img.height, one_predicted_boxes[:, 2] * img.height  # 1944
                one_predicted_boxes[:, 1], one_predicted_boxes[:, 3] = one_predicted_boxes[:,1] * img.width, one_predicted_boxes[:, 3] * img.width  # 2592

            else:  # for FNA MTL classifier, load boxes by image index
                one_ground_truth_boxes = ground_truth_boxes.item()[idx]
                one_predicted_boxes = prediction_boxes.item()[idx]
            print(one_ground_truth_boxes.shape, one_predicted_boxes.shape)

            if one_ground_truth_boxes.shape[0] > 0 or one_predicted_boxes.shape[0] > 0:
                # if there is at least one prediction or ground truth box in the image
                predict_polygon = connect_overlapped_boxes(one_predicted_boxes)
                ground_truth_polygon = connect_overlapped_boxes(one_ground_truth_boxes)
                # overlay polygon
                overlaid_img = self.overlay_polygons(img, ground_truth_polygon, (255,0,0), True)
                overlaid_img = self.overlay_polygons(overlaid_img, predict_polygon, (0,255,0), True)

                overlapped_polygon = predict_polygon.intersection(ground_truth_polygon)
                overlaid_img = self.overlay_polygons(overlaid_img, overlapped_polygon, (255,255, 0), True)
                save_filename = save_base_path + filename
                overlaid_img.save(save_filename)

                # calculate overlapped areas between two polygons
                pred_area_of_all_images = pred_area_of_all_images + predict_polygon.area
                gt_area_of_all_images = gt_area_of_all_images + ground_truth_polygon.area
                intersect_area_of_all_images = intersect_area_of_all_images + overlapped_polygon.area
                union_area_of_all_images = union_area_of_all_images + predict_polygon.union(ground_truth_polygon).area

        # ------- print statistics ------
        print(pred_area_of_all_images, gt_area_of_all_images, intersect_area_of_all_images, union_area_of_all_images)
        print('IOU', intersect_area_of_all_images/union_area_of_all_images)
        Precision = intersect_area_of_all_images / (pred_area_of_all_images)
        Recall = intersect_area_of_all_images / (gt_area_of_all_images)
        F1 = 2*Precision*Recall/(Precision+Recall)
        print('Precision:', Precision)
        print('Recall:', Recall)
        print('F1:', F1)


    def overlay_polygons(self, img, polygons, color, fill):
        overlaid_img = Image.new('RGBA', img.size, (*color, 0))

        if not polygons.is_empty:
            polygon_boundaries = polygons.boundary

            if polygons.boundary.geom_type == 'LineString':
                polygon_boundaries = [polygon_boundaries]
            else:
                polygon_boundaries = list(polygon_boundaries.geoms)

            for b in polygon_boundaries:  # combine two columns of x and y into tuples of x and y
                coords = np.dstack(b.coords.xy).tolist()
                coords = coords[0]
                coords = [tuple(x) for x in coords]  # convert list of lists to list of tuples

                draw = ImageDraw.Draw(overlaid_img)
                draw.line(coords, fill=color, width=10)
                if fill:
                    draw.polygon(coords, fill=(*color, 30))

        # Alpha composite these two images together to obtain the desired result.
        overlaid_img = Image.alpha_composite(img.convert('RGBA'), overlaid_img)

        return overlaid_img


    def overlay_two_model_overlapped_polygons_over_images(self, save_base_path, img_root_path, mask_names, list_of_ground_truth_polygons,
                                                          mtl_prediction_images_boxes, faster_rcnn_prediction_images_boxes, model_type, overlap_area_threshold, save_image_bool):
        '''
        In addition to drawing the intersection of predicitons from two models, 
        evaluate the performance in f1, precision and recall statistics
        '''
        if os.path.isdir(save_base_path) is False:
            os.makedirs(save_base_path)
        print('save_base_path', save_base_path)

        total_false_negative = 0
        total_false_positive = 0
        total_gt_overlaps = 0
        total_iou = 0
        iou_counter = 0

        overlapped_polygon_area_list = []

        for idx, filename in enumerate(mask_names):
            img_path = os.path.join(img_root_path, filename)
            img = Image.open(img_path)  # load images from paths

            one_ground_truth_polygons = list_of_ground_truth_polygons[idx]

            # one_ground_truth_polygons = ground_truth_boxes.item()[orig_filename]
            faster_rcnn_predicted_boxes = faster_rcnn_prediction_images_boxes[filename].copy()

            faster_rcnn_predicted_boxes[:, 0], faster_rcnn_predicted_boxes[:, 2] = faster_rcnn_predicted_boxes[:,0] *img.height, faster_rcnn_predicted_boxes[:,2] *img.height  # 1944
            faster_rcnn_predicted_boxes[:, 1], faster_rcnn_predicted_boxes[:, 3] = faster_rcnn_predicted_boxes[:,1] *img.width, faster_rcnn_predicted_boxes[:,3] *img.width  # 2592

            mtl_predicted_boxes = mtl_prediction_images_boxes[filename].copy()

            if not one_ground_truth_polygons.is_empty or mtl_predicted_boxes.shape[0] > 0 or faster_rcnn_predicted_boxes.shape[0] > 0:
                # if there is at least one prediction or ground truth box in the image
                
                # convert MTL boxes to polygons
                # convert Faster R-CNN boxes to polygons
                faster_rcnn_polygon = convert_boxes_to_polygons(faster_rcnn_predicted_boxes)
                mtl_polygon = convert_boxes_to_polygons(mtl_predicted_boxes)


                # overlap MTL polygons with Faster R-CNN polygons
                if 'MTL_faster-rcnn_overlap' in model_type:
                    overlapped_prediction_polygon = faster_rcnn_polygon.intersection(mtl_polygon)
                elif 'faster-rcnn_overlap' in model_type:
                    overlapped_prediction_polygon = faster_rcnn_polygon
                elif 'MTL_overlap' in model_type:
                    overlapped_prediction_polygon = mtl_polygon
                else:
                    raise ValueError('model type incorrect')
                
                # get list of areas from multipolygon to draw histogram (4/29/2023)
                overlapped_polygon_area_list = overlapped_polygon_area_list + get_multipolygon_area(overlapped_prediction_polygon)

                # overlap MTL + Faster R-CNN polygon with ground truth polygon
                overlapped_final_polygon = overlapped_prediction_polygon.intersection(one_ground_truth_polygons)

                # necessary because MTL prediction intersection with ground truth produces geometry Collection, instead of multipolygon
                if "GeometryCollection" == type(overlapped_final_polygon).__name__:
                    polygon_list = []
                    for line_or_polygon in overlapped_final_polygon.geoms:
                        # find polygons
                        if 'Polygon' == type(line_or_polygon).__name__:
                            polygon_list.append(line_or_polygon)

                    overlapped_final_polygon = MultiPolygon(polygon_list)  # convert to multipolygon

                # ------------------ IOU ---------------------
                if mtl_predicted_boxes.shape[0] > 0 and faster_rcnn_predicted_boxes.shape[0] > 0 and \
                    not one_ground_truth_polygons.is_empty and not overlapped_prediction_polygon.is_empty:
                    iou = calc_iou(one_ground_truth_polygons, overlapped_prediction_polygon)
                    total_iou = total_iou + iou
                    iou_counter = iou_counter + 1

                # ---------------- Count overlapped polygons --------------------
                gt_overlaps, false_negative, false_positive, gt_overlap_pair = count_overlap_polygons(one_ground_truth_polygons,
                                                                                                      overlapped_prediction_polygon,
                                                                                                      overlap_area_threshold)
                # if gt_overlaps > 0 or false_negative > 0 or false_positive > 0:
                # print(idx, filename, ' :  gt_overlaps', gt_overlaps, 'fn', false_negative, 'fp', false_positive)
                total_gt_overlaps = total_gt_overlaps + gt_overlaps
                total_false_negative = total_false_negative + false_negative
                total_false_positive = total_false_positive + false_positive

                # ------------------ overlay polygon over the image ----------------
                overlaid_img = self.overlay_polygons(img, one_ground_truth_polygons, (255, 0, 0), True)
                overlaid_img = self.overlay_polygons(overlaid_img, overlapped_prediction_polygon, (0, 255, 0), True)
                overlaid_img = self.overlay_polygons(overlaid_img, overlapped_final_polygon, (255, 255, 0), True)
                save_filename = save_base_path + filename
                if save_image_bool:
                    overlaid_img.save(save_filename)

            # evaluate by F1, Precision and Recall
        print('-------------------')
        print('model_type', model_type)
        print('overlap_area_threshold', overlap_area_threshold)
        print('tp:', total_gt_overlaps, 'fn:', total_false_negative, 'fp:', total_false_positive)
        if (total_gt_overlaps + total_false_positive) > 0:
            precision = total_gt_overlaps / (total_gt_overlaps + total_false_positive)
        else:
            precision = 0
        if (total_gt_overlaps + total_false_negative) > 0:
            recall = total_gt_overlaps / (total_gt_overlaps + total_false_negative)
        else:
            recall = 0

        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        if iou_counter > 0:
            avg_iou = total_iou/iou_counter
        else:
            avg_iou = 0
        print('precision:', round(precision, 3))
        print('recall:', round(recall, 3))
        print('f1:', round(f1, 3))
        print('total_iou:', round(avg_iou,3))

        # sklearn.metrics.matthews_corrcoef(y_true, y_pred)

        # if model_type == 'faster-rcnn_overlap':
        #     assert total_gt_overlaps == 16
        #     assert total_false_positive == 6
        #     assert total_false_negative == 21
        # elif model_type == 'MTL_faster-rcnn_overlap':
        #     assert total_gt_overlaps == 16
        #     assert total_false_positive == 3
        #     assert total_false_negative == 21

        
        return precision, recall, f1, avg_iou, overlapped_polygon_area_list


    def bounding_box_per_image_distribution(self, save_base_path, mask_names, ground_truth_boxes, predicted_boxes, model_type):
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


        plt.title(f'Ground Truth Vs. {model_type} Per Image', fontsize='x-large')
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


    def helper_plot_precision_recall_curve_at_thresholds(self, precision_list, recall_list, label_string):

        # sort them
        recall_sort_index = np.argsort(recall_list)
        precision_list = [precision_list[i] for i in recall_sort_index]
        recall_list = [recall_list[i] for i in recall_sort_index]

        lr_auc = auc(recall_list, precision_list)

        plt.plot(recall_list, precision_list, marker='.',
                label=f'{label_string}\nAUC={round(lr_auc, 3)}', lw=2)

    
    def plot_precision_recall_curve_at_thresholds(self, precision_list, recall_list, label_string, save_base_path):
        self.helper_plot_precision_recall_curve_at_thresholds(precision_list, recall_list, label_string)

        # plt.title('Slide Pass/Fail Precision-Recall curve', fontsize='x-large')
        plt.xlabel('Recall', fontsize='large')
        plt.ylabel('Precision', fontsize='large')

        plt.xlim(left=0, right=1.02)
        plt.ylim(bottom=0)
        plt.legend()
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)

        plt.savefig(save_base_path + f'precision_recall_curve_at_thresholds.svg')
        plt.close()


    def draw_histograms_of_mask_area_list(self, save_path, GT_polygon_area_list, mtl_polyon_area_list, faster_rcnn_polygon_area_list, fnanet_polyon_area_list):
        print("Number of areas and total area")
        print("GT", len(GT_polygon_area_list), sum(GT_polygon_area_list))
        print("faster", len(faster_rcnn_polygon_area_list), sum(faster_rcnn_polygon_area_list))
        print("MTL", len(mtl_polyon_area_list), sum(mtl_polyon_area_list))
        print("FNA-Net", len(fnanet_polyon_area_list), sum(fnanet_polyon_area_list))

        plt.figure()
        plt.hist(GT_polygon_area_list, bins=30, range=(0,300000))
        plt.savefig(f"{save_path}/manuscript/gt_polygon_area_list.png")
        plt.close()

        plt.figure()
        plt.hist(faster_rcnn_polygon_area_list, bins=30, range=(0,300000))
        plt.savefig(f"{save_path}/manuscript/faster_rcnn_polygon_area_list.png")
        plt.close()

        plt.figure()
        plt.hist(mtl_polyon_area_list, bins=30, range=(0,300000))
        plt.savefig(f"{save_path}/manuscript/mtl_polygon_area_list.png")
        plt.close()

        plt.figure()
        plt.hist(fnanet_polyon_area_list, bins=30, range=(0,300000))
        plt.savefig(f"{save_path}/manuscript/fnanet_polyon_area_list.png")
        plt.close()

    
    def manuscript_fig5_draw_comparison_bar_graph_with_errors(self, save_path, eval_type):
        # Figure 3 (c) in the manuscript
        # Precision, Recall and F1 score bar graphs from three different models: MTL, Faster R-CNN, and MTL + Faster R-CNN

        sns.set_theme(style="whitegrid", palette="muted")
        
        # ------------------------ f1 scores ------------------------
        # list_of_mean_list = [[0.1743, 0.3484, 0.6929, 0.8659, 0.7896, 0.9154],
        #         ['th6', 'th10', 'th6', 'th10', 'th6', 'th10'],
        #         ['RCNN', 'RCNN', 'MTL', 'MTL', 'Both', 'Both']]

        # # dicts with errors
        # mtl_error = {0.6929:0.140966224,0.8659:0.081491162}
        # faster_error = {0.1743:0.324696635, 0.3484:0.361580723}
        # mtl_faster_error = {0.7896:0.07916619, 0.9154:0.03404339}

        # ------------------------ AUC scores ------------------------
        list_of_mean_list = [[0.6494, 0.8963, 0.8221, 0.941, 0.8649, 0.9613],
                ['th6', 'th10', 'th6', 'th10', 'th6', 'th10'],
                ['RCNN', 'RCNN', 'MTL', 'MTL', 'Both', 'Both']]

        # dicts with errors
        faster_error = {0.6494:0.184641867, 0.8963:0.066892237}
        mtl_error = {0.8221:0.081846255,0.941:0.03881151}
        mtl_faster_error = {0.8649:0.080140353, 0.9613:0.027693045}
        # -----------------------------------------------------------

        # combine them; providing all the keys are unique
        z = {**mtl_error, **faster_error, **mtl_faster_error}

        transposed_list_of_list = np.array(list_of_mean_list).T.tolist()
        df = pd.DataFrame(transposed_list_of_list, columns=['val', 'metric', 'model'])
        df["val"] = pd.to_numeric(df["val"])

        fig, ax = plt.subplots(constrained_layout=True)

        sns.barplot(x="model", y="val", hue="metric", data=df)
        ax.set_ylim(0, 1.1)
        ax.legend(loc='upper left')
        
        for p in ax.patches:
            x = p.get_x()  # get the bottom left x corner of the bar
            w = p.get_width()  # get width of bar
            h = p.get_height()  # get height of bar
            error_offset_y = z[h]
            plt.vlines(x+w/2, h-error_offset_y, h+error_offset_y, color='k')  # draw a vertical line

        save_base_path = f"{save_path}manuscript/"
        if os.path.isdir(save_base_path) is False:
            os.makedirs(save_base_path)
        plt.savefig(f"{save_base_path}{eval_type}_fig5_bar_graph_comparison.svg")

    
    def manuscript_draw_comparison_bar_graph_with_errors(self, save_path, eval_type):
        # Figure 3 (c) in the manuscript
        # Precision, Recall and F1 score bar graphs from three different models: MTL, Faster R-CNN, and MTL + Faster R-CNN

        sns.set_theme(style="whitegrid", palette="muted")
        list_of_mean_list = [[0.118, 0.44, 0.73, 0.551, 0.307, 0.727, 0.432, 0.542, 0.318, 0.842, 0.432, 0.571],
                ['IOU', 'Precision', 'Recall', 'F1', 'IOU', 'Precision', 'Recall', 'F1', 'IOU', 'Precision', 'Recall', 'F1'],
                ['MTL', 'MTL', 'MTL', 'MTL', 'RCNN', 'RCNN', 'RCNN', 'RCNN', 'Both', 'Both', 'Both', 'Both']]
        
        list_of_mean_list = [[0.1486, 0.6080, 0.4500, 0.5058, 0.2497, 0.3276, 0.5007, 0.3552, 0.2902, 0.6874, 0.3008, 0.4131],
                ['IOU', 'Precision', 'Recall', 'F1', 'IOU', 'Precision', 'Recall', 'F1', 'IOU', 'Precision', 'Recall', 'F1'],
                ['MTL', 'MTL', 'MTL', 'MTL', 'RCNN', 'RCNN', 'RCNN', 'RCNN', 'Both', 'Both', 'Both', 'Both']]

        # dicts with errors
        iou_error = {0.1486:0.046824750243122874,0.2497:0.06494991166985277, 0.2902:0.09410608818186629}
        precision_error = {0.6080:0.22494560274141923, 0.3276:0.1983430418867766, 0.6874:0.25833208685863934}
        recall_error = {0.4500:0.18676425172882047, 0.5007:0.08396270727752463, 0.3008:0.13788650863512383}
        f1_error = {0.5058:0.1780586145725063, 0.3552:0.10321065268302565, 0.4131:0.1698763900930889}

        # combine them; providing all the keys are unique
        z = {**iou_error, **precision_error, **recall_error, **f1_error}

        transposed_list_of_list = np.array(list_of_mean_list).T.tolist()
        df = pd.DataFrame(transposed_list_of_list, columns=['val', 'metric', 'model'])
        df["val"] = pd.to_numeric(df["val"])

        fig, ax = plt.subplots(constrained_layout=True)

        sns.barplot(x="metric", y="val", hue="model", data=df)
        ax.set_ylim(0, 1.1)
        ax.legend(loc='upper left')
        
        for p in ax.patches:
            x = p.get_x()  # get the bottom left x corner of the bar
            w = p.get_width()  # get width of bar
            h = p.get_height()  # get height of bar
            error_offset_y = z[h]
            plt.vlines(x+w/2, h-error_offset_y, h+error_offset_y, color='k')  # draw a vertical line

        save_base_path = f"{save_path}manuscript/"
        if os.path.isdir(save_base_path) is False:
            os.makedirs(save_base_path)
        plt.savefig(f"{save_base_path}{eval_type}_bar_graph_comparison.svg")


    def manuscript_draw_MTL_comparison_bar_graph_with_errors(self, save_path, eval_type, summary_eval_dict):
        # Figure 3 (a) in the manuscript
        # Precision, Recall and F1 score bar graphs from three different models: MTL, Faster R-CNN, and MTL + Faster R-CNN
        sns.set_theme(style="whitegrid", palette="muted")

        # get error bar
        z = {}
        list_of_mean_list = [[],[],[]]
        for mtl_model_type in summary_eval_dict.keys():
            for a_metric in ['precision', 'recall', 'f1']:
                a_mean = np.round(np.mean(summary_eval_dict[mtl_model_type][a_metric]), 8)
                a_std = np.round(np.std(summary_eval_dict[mtl_model_type][a_metric]), 8)
                
                z[a_mean] = a_std
                list_of_mean_list[0].append(a_mean)
                list_of_mean_list[1].append(a_metric)
                list_of_mean_list[2].append(mtl_model_type)

        transposed_list_of_list = np.array(list_of_mean_list).T.tolist()
        df = pd.DataFrame(transposed_list_of_list, columns=['val', 'metric', 'model'])
        df["val"] = pd.to_numeric(df["val"])
        
        fig, ax = plt.subplots(constrained_layout=True)

        # sns.barplot(x="model", y="val", hue="metric", data=df, order=['classifier', 'MTL_auto_aut_seg', 'MTL_auto_reg_seg', 'MTL_auto_reg_aut', 'MTL_cls1_reg0_aut0_seg0.75', 'MTL_auto_reg', 'MTL_auto'])
        # sns.barplot(x="metric", y="val", hue="model", data=df, hue_order=['classifier', 'MTL_auto_aut_seg', 'MTL_auto_reg_seg', 'MTL_auto_reg_aut', 'MTL_cls1_reg0_aut0_seg0.75', 'MTL_auto_reg', 'MTL_auto'])
        sns.barplot(x="metric", y="val", hue="model", data=df, hue_order=['MTL_auto','classifier', 'MTL_auto_reg_seg', 'MTL_auto_aut_seg', 'MTL_auto_reg', 'MTL_cls1_reg0_aut0_seg0.75', 'MTL_auto_reg_aut'])
        ax.set_ylim(0, 1)
        ax.legend(loc='upper left')
        
        for p in ax.patches:
            x = p.get_x()  # get the bottom left x corner of the bar
            w = p.get_width()  # get width of bar
            h = p.get_height()  # get height of bar
            error_offset_y = z[h]
            plt.vlines(x+w/2, h-error_offset_y, h+error_offset_y, color='k')  # draw a vertical line

        save_base_path = f"{save_path}manuscript/"
        if os.path.isdir(save_base_path) is False:
            os.makedirs(save_base_path)
        plt.savefig(f"{save_base_path}{eval_type}_bar_graph_MTL_comparison.svg")
