'''
Author Junbong Jang
Creation Date: 10/28/2020

Includes functions for evaluation such as counting overlaps between ground truth and prediction boxes
Calculating the overlapped area between boxes, polygons and converting boxes or masks to polygons
Many of these functions are used in conjunction with the visualizer.py
'''
import numpy as np
from shapely.ops import unary_union
import geopandas as gpd
from shapely.geometry import Polygon


def calculate_polygon_area(ordered_coordinates):
    x_y = ordered_coordinates.reshape(-1, 2)

    x = x_y[:, 0]
    y = x_y[:, 1]

    # shoelace formula
    S1 = np.dot(x, np.roll(y, -1))
    S2 = np.dot(y, np.roll(x, -1))
    area = .5 * np.absolute(S1 - S2)

    return area


def get_multipolygon_area(a_multipolygon):
    if a_multipolygon.geom_type == 'Polygon':
        processed_multipolygon = [a_multipolygon]
    else:
        processed_multipolygon = list(a_multipolygon.geoms)

    area_list = []
    for a_polygon in processed_multipolygon:
        area_list.append(a_polygon.area)

    return area_list


def calculate_box_overlap_area(box1, box2):
    # https://stackoverflow.com/questions/27152904/calculate-overlapped-area-between-two-rectangles
    ymin1, xmin1, ymax1, xmax1 = box1[0], box1[1], box1[2], box1[3]
    ymin2, xmin2, ymax2, xmax2 = box2[0], box2[1], box2[2], box2[3]
    dx = min(xmax1, xmax2) - max(xmin1, xmin2)
    dy = min(ymax1, ymax2) - max(ymin1, ymin2)

    area = 0
    if (dx >= 0) and (dy >= 0):  # negative if they don't overlap
        area = dx * dy

    return area


def calculate_box_area(box):
    ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]
    return (xmax-xmin) * (ymax-ymin)


def convert_boxes_to_polygons(boxes, is_union_boxes=True):
    # connect boxes to form a big polygon
    def convert_box_coordinate_to_polygon(box):
        # from box coordinates ymin, xmin, ymax, xmax to (x1,y1), (x2,y2), (x3,y3), (x4,y4)
        ymin, xmin, ymax, xmax = box
        x1, y1 = xmin, ymin
        x2, y2 = xmax, ymin
        x3, y3 = xmax, ymax
        x4, y4 = xmin, ymax
        return Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])

    polygons = []
    for box in boxes:
        polygon = convert_box_coordinate_to_polygon(box)
        polygons.append(polygon)

    if is_union_boxes:
        polygons = gpd.GeoSeries(unary_union(polygons))[0]  # connect_overlapped_boxes

    # below is not necessary because shapely polygon already figures out which polygons are interiors with unary_union
    # if polygons.boundary.geom_type == 'MultiLineString':
    #     print(len(polygons.boundary))
    #     new_polygons = find_interior_polygons_of_a_polygon(polygons)
        # subtract operation between big polygon and small polygon within
        # And operation above results

    return polygons


def find_interior_polygons_of_a_polygon(polygons):
    # see whether one polygon contains another polygon
    new_polygons = []
    num_polygons = len(polygons)
    for i, polygon in enumerate(polygons):
        for j in range(num_polygons):
            if i == j:
                if polygon.contains(polygons[j]):
                    polygon = polygon.difference(polygons[j])
        new_polygons.append(polygon)

    return new_polygons


def count_polygons(polygon_obj):
    total_count = 0
    if not polygon_obj.is_empty:
        if polygon_obj.geom_type == 'MultiPolygon':
            total_count = len(polygon_obj)
        elif polygon_obj.geom_type == 'Polygon':
            total_count = 1
    return total_count


def count_boxes(image_names, input_boxes):
    boxes_total = 0
    for image_name in image_names:
        boxes = input_boxes.item()[image_name]
        boxes_total = boxes_total + len(boxes)
    return boxes_total


def count_overlap_box(ground_truth_boxes, predicted_boxes):
    total_gt_boxes = ground_truth_boxes.shape[0]
    gt_overlaps = 0
    gt_overlap_pair = []
    # For every ground truth box against all prediction boxes
    for i, ground_truth_box in enumerate(ground_truth_boxes):
        for j, predicted_box in enumerate(predicted_boxes):
            overlapped_area = calculate_box_overlap_area(ground_truth_box, predicted_box)
            # If overlapped area is greater than 50% of ground truth or prediction box
            if overlapped_area > calculate_box_area(ground_truth_box) * 0.5 or overlapped_area > calculate_box_area(predicted_box) * 0.5:
                # For one prediction, count all ground truth inside it
                # For one ground truth, count only once for prediction boxes inside it
                gt_overlaps = gt_overlaps + 1
                gt_overlap_pair.append((i,j))
                break

    not_fp_pair = []
    # to count false_positive
    false_positive = 0
    for i, predicted_box in enumerate(predicted_boxes):
        false_positive_flag = True
        for j, ground_truth_box in enumerate(ground_truth_boxes):
            overlapped_area = calculate_box_overlap_area(ground_truth_box, predicted_box)
            if overlapped_area > calculate_box_area(ground_truth_box) * 0.5 or overlapped_area > calculate_box_area(predicted_box) * 0.5:
                false_positive_flag = False
                not_fp_pair.append((i,j))
                break
        if false_positive_flag:
            false_positive = false_positive + 1

    # print(gt_overlap_pair)
    # print(not_fp_pair)
    # print('------------------')

    # print('total_gt_boxes', total_gt_boxes)
    # print('overlaps', gt_overlaps)
    false_negative = total_gt_boxes - gt_overlaps

    return gt_overlaps, false_negative, false_positive, gt_overlap_pair


def calc_iou(ground_truth_polygons, predicted_polygons):
    gt_union = unary_union(ground_truth_polygons)
    predicted_union = unary_union(predicted_polygons)
    a_intersection = gt_union.intersection(predicted_union).area
    a_union = gt_union.union(predicted_union).area

    iou = a_intersection / a_union

    return iou


def count_overlap_polygons(ground_truth_polygons, predicted_polygons, overlap_area_threshold):
    if ground_truth_polygons.geom_type == 'Polygon':
        total_gt_polygons = 1
        ground_truth_polygons = [ground_truth_polygons]
    else:
        ground_truth_polygons = list(ground_truth_polygons.geoms)
        total_gt_polygons = len(ground_truth_polygons)

    if predicted_polygons.geom_type == 'Polygon':
        predicted_polygons = [predicted_polygons]
    else:
        predicted_polygons = list(predicted_polygons.geoms)


    gt_overlaps = 0
    gt_overlap_pair = []
    total_overlapped_area = 0
    # overlap_area_threshold = 0.1  # 0.5 origianlly but 0.1 is to count the predicted small box inside ground truth for final_eval
    # For every ground truth box against all prediction polygons
    for i, ground_truth_polygon in enumerate(ground_truth_polygons):
        for j, predicted_polygon in enumerate(predicted_polygons):
            overlapped_area = ground_truth_polygon.intersection(predicted_polygon).area
            total_overlapped_area = total_overlapped_area + overlapped_area
            # If overlapped area is greater than 50% of ground truth or prediction polygon
            if overlapped_area > ground_truth_polygon.area * overlap_area_threshold or overlapped_area > predicted_polygon.area * overlap_area_threshold:
                # For one prediction, count all ground truth inside it
                # For one ground truth, count only once for prediction polygons inside it
                gt_overlaps = gt_overlaps + 1
                gt_overlap_pair.append((i,j))
                break

    # to count false_positive
    not_fp_pair = []
    false_positive = 0
    for i, predicted_polygon in enumerate(predicted_polygons):
        false_positive_flag = True
        for j, ground_truth_polygon in enumerate(ground_truth_polygons):
            overlapped_area = ground_truth_polygon.intersection(predicted_polygon).area
            if overlapped_area > ground_truth_polygon.area * overlap_area_threshold or overlapped_area > predicted_polygon.area * overlap_area_threshold:
                false_positive_flag = False
                not_fp_pair.append((i,j))
                break
        if false_positive_flag:
            false_positive = false_positive + 1

    false_negative = total_gt_polygons - gt_overlaps

    return gt_overlaps, false_negative, false_positive, gt_overlap_pair


