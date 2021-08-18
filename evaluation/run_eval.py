'''
Author Junbong Jang
5/8/2021

To run bootstrapping and visualize box images
'''
import os
import numpy as np
from tqdm import tqdm

from visualizer import Visualizer
import bootstrapping as bootstrap
import bootstrapping_visualization as bootstrap_viz
from explore_data import get_files_in_folder, get_images_by_subject, print_images_by_subject_statistics
import pandas as pd
from PIL import Image


def bootstrap_data(test_image_names, model_type, bootstrap_repetition_num, load_path, save_base_path, img_root_path):

    np_predicted_detection_boxes = np.load(f"{load_path}/{model_type}_boxes.npy", allow_pickle=True)
    np_ground_truth_boxes = np.load(f"{load_path}/ground_truth_boxes.npy", allow_pickle=True)

    print('Data Organization')
    test_images_by_subject = get_images_by_subject(test_image_names)
    print_images_by_subject_statistics(test_images_by_subject)

    # Follicular Cluster Counting after bootstrapping
    img_path = os.path.join(img_root_path, test_image_names[0])
    img = Image.open(img_path)  # load images from paths
    img_size = {}
    img_size['height'], img_size['width'] = img.size
    bootstrapped_box_counts_df, bootstrapped_area_df = bootstrap.bootstrap_box_counts(test_image_names, test_images_by_subject, model_type, img_size,
                                                        np_ground_truth_boxes, np_predicted_detection_boxes, bootstrap_repetition_num)

    # box_counts_df.to_excel('generated/bootstrapped_box_counts.xlsx', index=False)
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
        a_precision, a_recall, a_f1 = bootstrap.stats_at_threshold(bootstrapped_df, ground_truth_min_follicular,
                                                         predicted_min_follicular, DEBUG=True)
        predicted_min_follicular_list.append(predicted_min_follicular)
        precision_list.append(a_precision)
        recall_list.append(a_recall)
        f1_list.append(a_f1)

    bootstrap_viz.plot_precision_recall_curve_at_thresholds(y_true, precision_list, recall_list, save_base_path)
    bootstrap_viz.plot_performance_at_thresholds(predicted_min_follicular_list, precision_list, recall_list, f1_list,
                                   save_base_path)


def convert_mask_to_box(ground_truth_mask_root_path):
    ground_truth_mask_names = [file for file in os.listdir(ground_truth_mask_root_path) if file.endswith(".png")]
    ground_truth_mask_names.sort()

    visualizer_obj = Visualizer()
    visualizer_obj.mask_to_box('generated/', ground_truth_mask_root_path, ground_truth_mask_names, 'ground_truth_3cat')


def run_visualize_images(load_path, model_type, img_root_path, ground_truth_mask_root_path):

    ground_truth_mask_names = [file for file in os.listdir(ground_truth_mask_root_path) if file.endswith(".png")]
    ground_truth_mask_names.sort()

    predicted_detection_boxes = np.load(f"{load_path}/{model_type}_boxes.npy", allow_pickle=True)
    ground_truth_boxes = np.load(f"{load_path}/ground_truth_boxes.npy", allow_pickle=True)
    visualizer_obj = Visualizer()

    # -------------------- Boxed Images Visualization -----------------------------
    # save_base_path = f"{load_path}/{model_type}_boxes/"
    # if os.path.isdir(save_base_path) is False:
    #     os.mkdir(save_base_path)
    # print('save_base_path', save_base_path)
    #
    # visualizer_obj.overlay_boxes_over_images(save_base_path, img_root_path, ground_truth_mask_names, ground_truth_boxes,
    #                                 predicted_detection_boxes, model_type=model_type)
    # bounding_box_per_image_distribution(save_base_path, ground_truth_mask_names, ground_truth_boxes, faster_rcnn_boxes, model_type=model_type)

    # -------------------- Polygon Visualization -----------------------------
    save_base_path = f"{load_path}/{model_type}_polygon/"
    if os.path.isdir(save_base_path) is False:
        os.mkdir(save_base_path)
    print('save_base_path', save_base_path)

    visualizer_obj.overlay_polygons_over_images(save_base_path, img_root_path, ground_truth_mask_names, ground_truth_boxes,
                                    predicted_detection_boxes, model_type=model_type)


def run_cam(model_type, img_root_path):
    predicted_feature_maps = np.load(f"generated/{model_type}_features.npy", allow_pickle=True)
    # feature_name = 'rpn_features_to_crop'  # 40x40x1088
    feature_name= 'rpn_box_predictor_features'  # 40x40x512
    save_heatmap_path = f'generated/{model_type}_boxes/{feature_name}/'
    if os.path.isdir(save_heatmap_path) is False:
        os.mkdir(save_heatmap_path)
    visualizer_obj = Visualizer()
    for image_name in tqdm(predicted_feature_maps.item().keys()):
        feature_map = predicted_feature_maps.item()[image_name][feature_name]
        visualizer_obj.visualize_feature_activation_map(feature_map, img_root_path, image_name, save_heatmap_path)


def get_data_path(model_type):
    if 'faster' in model_type:
        base_path = 'C:/Users/Jun/Documents/PycharmProjects/MARS-Net/'
        ground_truth_mask_root_path = base_path + 'assets/FNA/FNA_test/mask_processed/'
        img_root_path = base_path + 'assets/FNA/FNA_test/img/'
        # img_root_path = base_path + 'FNA/assets/all-patients/img/'
        # img_root_path = base_path + 'tensorflowAPI/research/object_detection/dataset_tools/assets/images_test/'
        load_path = 'C:/Users/Jun/Documents/PycharmProjects/FNA/generated/'
    else:
        base_path = 'C:/Users/Jun/Documents/PycharmProjects/MARS-Net/'
        # ground_truth_mask_root_path = base_path + 'assets/FNA/FNA_valid_fold0/mask_processed/'
        # img_root_path = base_path + 'assets/FNA/FNA_valid_fold0/img/'
        # load_path = base_path + 'models/results/predict_wholeframe_round1_FNA_VGG19_MTL_auto_reg_aut_input256/FNA_valid_fold0/frame2_A_repeat0/'
        ground_truth_mask_root_path = base_path + 'assets/FNA/FNA_test/mask_processed/'
        img_root_path = base_path + 'assets/FNA/FNA_test/img/'
        load_path = base_path + 'models/results/predict_wholeframe_round1_FNA_VGG19_MTL_auto_reg_aut_input256_patience_10/FNA_test/frame2_training_repeat0/'

    save_base_path = f'{load_path}/bootstrapped_{model_type}/'
    if os.path.isdir(save_base_path) is False:
        os.mkdir(save_base_path)

    return ground_truth_mask_root_path, img_root_path, load_path, save_base_path


def run_eval(model_type, ground_truth_mask_root_path, img_root_path, load_path, save_base_path):
    test_image_names = get_files_in_folder(img_root_path)
    test_image_names.sort()

    # bootstrap_data(test_image_names, model_type, 10000, load_path, save_base_path, img_root_path)

    # bootstrapped_box_counts_df = pd.read_csv(f'{save_base_path}bootstrapped_box_counts_df.csv', header=None)
    # ground_truth_min_follicular = int(bootstrapped_box_counts_df[0].mean())  # 100  # 15
    # bootstrap_analysis(bootstrapped_box_counts_df, test_image_names, ground_truth_min_follicular, save_base_path)

    bootstrapped_area_df = pd.read_csv(f'{save_base_path}bootstrapped_area_df.csv', header=None)
    ground_truth_mean_area = int(bootstrapped_area_df[0].mean())
    bootstrap_analysis(bootstrapped_area_df, test_image_names, ground_truth_mean_area, save_base_path)

    # run_visualize_images(load_path, model_type, img_root_path, ground_truth_mask_root_path)
    # run_cam(model_type, img_root_path)


if __name__ == "__main__":
    # model_type = 'faster_640'  # 'faster_640_stained_improved'
    model_type = 'MTL_auto_reg_aut'

    ground_truth_mask_root_path, img_root_path, load_path, save_base_path = get_data_path(model_type)
    run_eval(model_type, ground_truth_mask_root_path, img_root_path, load_path, save_base_path)