'''
Author Junbong Jang
Creation Date: 10/28/2020

Count overlaps between ground truth and prediction boxes
'''


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