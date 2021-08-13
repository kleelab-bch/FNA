'''
Author Junbong Jang
Creation Date: 10/28/2020

Count overlaps between ground truth and prediction boxes
'''


def calculate_polygon_area(ordered_coordinates):
    x_y = ordered_coordinates.reshape(-1, 2)

    x = x_y[:, 0]
    y = x_y[:, 1]

    # shoelace formula
    S1 = np.dot(x, np.roll(y, -1))
    S2 = np.dot(y, np.roll(x, -1))
    area = .5 * np.absolute(S1 - S2)

    return area


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


from shapely.ops import cascaded_union
import geopandas as gpd
from shapely.geometry import Polygon
def connect_overlapped_boxes(boxes):
    # connect boxes to form a big polygon
    def convert_box_coordinate_to_polygon(box):
        # from box coordinates ymin, xmin, ymax, xmax to (x1,y1), (x2,y2), (x3,y3), (x4,y4)
        ymin, xmin, ymax, xmax = box
        x1, y1 = xmin, ymin
        x2, y3 = xmax, ymin
        x3, y3 = xmax, ymax
        x4, y4 = xmin, ymax
        return Polygon([(x1, y1), (x2, y3), (x3, y3), (x4, y4)])

    polygons = []
    for box in boxes:
        polygon = convert_box_coordinate_to_polygon(box)
        polygons.append(polygon)

    boundary = gpd.GeoSeries(cascaded_union(polygons))

    return boundary


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