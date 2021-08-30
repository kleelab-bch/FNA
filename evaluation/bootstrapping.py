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
from tqdm import tqdm
from PIL import Image
import os

from bootstrapping_visualization import *
from calc_polygon import convert_boxes_to_polygons
import bootstrapping_visualization as bootstrap_viz
from explore_data import get_images_by_subject, print_images_by_subject_statistics


def bootstrap_data(test_image_names, model_type, bootstrap_repetition_num, np_predicted_detection_boxes, np_ground_truth_boxes, save_base_path, img_root_path):
    # bootstrap both ground truth and prediction boxes
    print('Data Organization')
    test_images_by_subject = get_images_by_subject(test_image_names)
    print_images_by_subject_statistics(test_images_by_subject)

    # bootstrap data and count Follicular Clusters in each bootstrap sample
    img_path = os.path.join(img_root_path, test_image_names[0])
    img = Image.open(img_path)  # load images from paths
    img_size = {}
    img_size['height'], img_size['width'] = img.size
    bootstrapped_box_counts_df, bootstrapped_area_df = bootstrap_box_counts_area(test_image_names, test_images_by_subject, model_type, img_size,
                                                        np_ground_truth_boxes, np_predicted_detection_boxes, bootstrap_repetition_num)

    print('bootstrapped data shape: ', bootstrapped_box_counts_df.shape, bootstrapped_area_df.shape)
    print()

    bootstrapped_box_counts_df.to_csv(f'{save_base_path}bootstrapped_box_counts_df.csv', index=False, header=False)
    bootstrapped_area_df.to_csv(f'{save_base_path}bootstrapped_area_df.csv',index=False, header=False)

    return test_image_names, save_base_path


def bootstrap_analysis(bootstrapped_df, test_image_names, ground_truth_min_follicular, save_base_path):
    print('bootstrap_analysis', ground_truth_min_follicular)
    # Data Exploration
    bootstrap_viz.plot_histogram(bootstrapped_df, test_image_names, save_base_path)
    bootstrap_viz.plot_scatter(bootstrapped_df, ground_truth_min_follicular, save_base_path)

    # Roc curve tutorials
    y_true = bootstrapped_df[0] >= ground_truth_min_follicular
    # y_pred = bootstrapped_df[1] >= predicted_min_follicular
    # plot_roc_curve(y_true, y_pred)
    # plot_precision_recall_curve(y_true, y_pred)

    # -------- Varying Predicted Min Follicular Thresholds ------------
    precision_list = []
    recall_list = []
    f1_list = []
    predicted_min_follicular_list = []
    for_step = 1
    if len(str(ground_truth_min_follicular)) > 2:
        for_step = 10**(len(str(int(ground_truth_min_follicular)))-2)
    print('for_step', for_step)
    for predicted_min_follicular in range(0, ground_truth_min_follicular * 2, for_step):
        a_precision, a_recall, a_f1 = stats_at_threshold(bootstrapped_df, ground_truth_min_follicular,
                                                         predicted_min_follicular, DEBUG=True)
        predicted_min_follicular_list.append(predicted_min_follicular)
        precision_list.append(a_precision)
        recall_list.append(a_recall)
        f1_list.append(a_f1)

    bootstrap_viz.plot_precision_recall_curve_at_thresholds(y_true, precision_list, recall_list, save_base_path)
    bootstrap_viz.plot_performance_at_thresholds(predicted_min_follicular_list, precision_list, recall_list, f1_list,
                                   save_base_path)


def bootstrap_box_counts_area(image_names, images_by_subject, model_type, img_size, np_ground_truth_boxes, np_prediction_boxes, bootstrap_repetition_num):
    '''
    Count the total number of boxes per object detected image
    We assume each box represents one follicular cluster detection.
    Secretion/Artifact boxes should have been processed and removed beforehand.

    Hierachical bootstrapping if images_by_subject is not None

    :param image_names:
    :param images_by_subject:
    :param np_ground_truth_boxes:
    :param np_prediction_boxes:
    :param bootstrap_repetition_num:
    :return:
    '''
    image_name_to_index = {}
    for i, image_name in enumerate(image_names):
        image_name_to_index[image_name] = i
    testset_sample_size = len(image_names)

    box_counts = np.zeros(shape=(bootstrap_repetition_num, 2))
    bootstrapped_areas = np.zeros(shape=(bootstrap_repetition_num, 2))
    for bootstrap_repetition_index in tqdm(range(bootstrap_repetition_num)):

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
        prediction_boxes_total = 0
        ground_truth_area_total = 0
        prediction_area_total = 0

        for chosen_image_name in bootstrap_sampled_image_names:
            img_index = image_name_to_index[chosen_image_name]
            if 'faster' in model_type:
                ground_truth_boxes = np_ground_truth_boxes.item()[chosen_image_name]
                prediction_boxes = np_prediction_boxes.item()[chosen_image_name]
                prediction_boxes = prediction_boxes.astype('float64')
                prediction_boxes[:, 0], prediction_boxes[:, 2] = prediction_boxes[:, 0] * img_size['height'], prediction_boxes[:, 2] * img_size['height'] # 1944
                prediction_boxes[:, 1], prediction_boxes[:, 3] = prediction_boxes[:, 1] * img_size['width'], prediction_boxes[:,3] * img_size['width']  # 2592

            else:
                ground_truth_boxes = np_ground_truth_boxes.item()[img_index]
                prediction_boxes = np_prediction_boxes.item()[img_index]

            ground_truth_boxes_total = ground_truth_boxes_total + len(ground_truth_boxes)
            ground_truth_polygon = convert_boxes_to_polygons(ground_truth_boxes)
            if not ground_truth_polygon.is_empty:
                ground_truth_area_total = ground_truth_area_total + ground_truth_polygon.area

            # count model prediction boxes
            prediction_boxes_total = prediction_boxes_total + len(prediction_boxes)
            predict_polygon = convert_boxes_to_polygons(prediction_boxes)
            if not predict_polygon.is_empty:
                prediction_area_total = prediction_area_total + predict_polygon.area

        box_counts[bootstrap_repetition_index, :] = ground_truth_boxes_total, prediction_boxes_total
        bootstrapped_areas[bootstrap_repetition_index, :] = ground_truth_area_total, prediction_area_total

    box_counts_df = pd.DataFrame(box_counts)
    bootstrapped_area_df = pd.DataFrame(bootstrapped_areas)

    return box_counts_df, bootstrapped_area_df


def bootstrap_box_polygon(image_names, images_by_subject, model_type, img_size, np_ground_truth_boxes, np_prediction_boxes, bootstrap_repetition_num):
    '''
    Count the total number of boxes per object detected image
    We assume each box represents one follicular cluster detection.
    Secretion/Artifact boxes should have been processed and removed beforehand.

    Hierachical bootstrapping if images_by_subject is not None

    :param image_names:
    :param images_by_subject:
    :param np_ground_truth_boxes:
    :param np_prediction_boxes:
    :param bootstrap_repetition_num:
    :return:
    '''
    image_name_to_index = {}
    for i, image_name in enumerate(image_names):
        image_name_to_index[image_name] = i
    testset_sample_size = len(image_names)

    box_counts = np.zeros(shape=(bootstrap_repetition_num, 2))
    bootstrapped_areas = np.zeros(shape=(bootstrap_repetition_num, 2))
    for bootstrap_repetition_index in tqdm(range(bootstrap_repetition_num)):

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
        prediction_boxes_total = 0
        ground_truth_area_total = 0
        prediction_area_total = 0

        for chosen_image_name in bootstrap_sampled_image_names:
            img_index = image_name_to_index[chosen_image_name]
            if 'faster' in model_type:
                ground_truth_boxes = np_ground_truth_boxes.item()[chosen_image_name]
                prediction_boxes = np_prediction_boxes.item()[chosen_image_name]
                prediction_boxes = prediction_boxes.astype('float64')
                prediction_boxes[:, 0], prediction_boxes[:, 2] = prediction_boxes[:, 0] * img_size['height'], prediction_boxes[:, 2] * img_size['height'] # 1944
                prediction_boxes[:, 1], prediction_boxes[:, 3] = prediction_boxes[:, 1] * img_size['width'], prediction_boxes[:,3] * img_size['width']  # 2592

            else:
                ground_truth_boxes = np_ground_truth_boxes.item()[img_index]
                prediction_boxes = np_prediction_boxes.item()[img_index]

            ground_truth_boxes_total = ground_truth_boxes_total + len(ground_truth_boxes)
            ground_truth_polygon = convert_boxes_to_polygons(ground_truth_boxes)
            if not ground_truth_polygon.is_empty:
                ground_truth_area_total = ground_truth_area_total + ground_truth_polygon.area

            # count model prediction boxes
            prediction_boxes_total = prediction_boxes_total + len(prediction_boxes)
            predict_polygon = convert_boxes_to_polygons(prediction_boxes)
            if not predict_polygon.is_empty:
                prediction_area_total = prediction_area_total + predict_polygon.area

        box_counts[bootstrap_repetition_index, :] = ground_truth_boxes_total, prediction_boxes_total
        bootstrapped_areas[bootstrap_repetition_index, :] = ground_truth_area_total, prediction_area_total

    box_counts_df = pd.DataFrame(box_counts)
    bootstrapped_area_df = pd.DataFrame(bootstrapped_areas)

    return box_counts_df, bootstrapped_area_df


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