"""
Author Junbong Jang
Date: 7/29/2020

Bootstrap sample the test set images
Load ground truth and faster rcnn boxes and count them.
After sampling distribution is obtained, to calculate confidence intervals

Refered to https://machinelearningmastery.com/a-gentle-introduction-to-the-bootstrap-method/
https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/
"""

import pandas as pd
from sklearn.utils import resample
from bootstrapping_visualization import *
import os


def bootstrap_box_counts(image_names, images_by_subject, np_ground_truth_boxes, np_model_prediction_boxes, bootstrap_repetition_num):
    '''
    Count the total number of boxes per object detected image
    We assume each box represents one follicular cluster detection.
    Secretion/Artifact boxes should have been processed and removed beforehand.

    Hierachical bootstrapping if images_by_subject is not None

    :param image_names:
    :param images_by_subject:
    :param np_ground_truth_boxes:
    :param np_model_prediction_boxes:
    :param bootstrap_repetition_num:
    :return:
    '''
    image_name_to_index = {}
    for i, image_name in enumerate(image_names):
        image_name_to_index[image_name] = i
    testset_sample_size = len(image_names)
    testset_indices = np.arange(0, testset_sample_size, 1)

    box_counts = np.zeros(shape=(bootstrap_repetition_num, 2))
    for bootstrap_repetition_index in range(bootstrap_repetition_num):

        # ------- bootstrap subjects ------
        if images_by_subject != None:
            bootstrap_sampled_subjects = resample(list(images_by_subject.keys()), replace=True, n_samples=len(images_by_subject.keys()),
                                                 random_state=bootstrap_repetition_index)
            # only get images from sampled subjects
            image_names = []
            for bootstrap_sampled_subject in bootstrap_sampled_subjects:
                image_names = image_names + images_by_subject[bootstrap_sampled_subject]

        # ------- bootstrap images ---------
        bootstrap_sampled_image_names = resample(image_names, replace=True, n_samples=testset_sample_size,
                                             random_state=bootstrap_repetition_index)

        ground_truth_boxes_total = 0
        model_prediction_boxes_total = 0
        for chosen_image_name in bootstrap_sampled_image_names:
            img_index = image_name_to_index[chosen_image_name]  # do this, enumerate() do not work
            # count ground truth boxes
            ground_truth_boxes = np_ground_truth_boxes.item()[img_index]  # chosen_image
            ground_truth_boxes_total = ground_truth_boxes_total + len(ground_truth_boxes)

            # count faster rcnn boxes
            model_prediction_boxes = np_model_prediction_boxes.item()[img_index]  # chosen_image
            model_prediction_boxes_total = model_prediction_boxes_total + len(model_prediction_boxes)


        box_counts[bootstrap_repetition_index, :] = ground_truth_boxes_total, model_prediction_boxes_total

    box_counts_df = pd.DataFrame(box_counts)
    return box_counts_df


def count_boxes(image_names, input_boxes):
    boxes_total = 0
    for image_name in image_names:
        boxes = input_boxes.item()[image_name]
        boxes_total = boxes_total + len(boxes)
    return boxes_total


def stats_at_threshold(box_counts_df, ground_truth_min_follicular, predicted_min_follicular, DEBUG):
    true_positive = box_counts_df.loc[
        (box_counts_df[0] >= ground_truth_min_follicular) & (box_counts_df[1] >= predicted_min_follicular)]
    true_negative = box_counts_df.loc[
        (box_counts_df[0] < ground_truth_min_follicular) & (box_counts_df[1] < predicted_min_follicular)]
    false_positive = box_counts_df.loc[
        (box_counts_df[0] < ground_truth_min_follicular) & (box_counts_df[1] >= predicted_min_follicular)]
    false_negative = box_counts_df.loc[
        (box_counts_df[0] >= ground_truth_min_follicular) & (box_counts_df[1] < predicted_min_follicular)]

    true_positive = len(true_positive)
    true_negative = len(true_negative)
    false_positive = len(false_positive)
    false_negative = len(false_negative)

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    F1 = 2 * (precision * recall) / (precision + recall)

    if DEBUG:
        print('pred_min_follicular:', predicted_min_follicular)
        print('true_positives', true_positive, end='  ')
        print('true_negative', true_negative, end='  ')
        print('false_positives', false_positive, end='  ')
        print('false_negative', false_negative)

        print('precision', precision, end='  ')
        print('recall', recall, end='  ')
        print('F1', F1)

    return precision, recall, F1