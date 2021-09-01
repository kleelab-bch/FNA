'''
Author Junbong Jang
5/8/2021

To run bootstrapping and visualize box images
'''
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image

from calc_polygon import convert_boxes_to_polygons, count_overlap_polygons
from visualizer import Visualizer
from explore_data import get_files_in_folder, get_images_by_subject, print_images_by_subject_statistics
from bootstrapping import bootstrap_analysis, bootstrap_data, bootstrap_two_model_polygons


def convert_mask_to_box(ground_truth_mask_root_path):
    ground_truth_mask_names = [file for file in os.listdir(ground_truth_mask_root_path) if file.endswith(".png")]
    ground_truth_mask_names.sort()

    visualizer_obj = Visualizer()
    visualizer_obj.mask_to_box('generated/', ground_truth_mask_root_path, ground_truth_mask_names, 'ground_truth_3cat')


def convert_mask_to_polygon(ground_truth_mask_root_path):
    # return list of polygons
    ground_truth_mask_names = [file for file in os.listdir(ground_truth_mask_root_path) if file.endswith(".png")]
    ground_truth_mask_names.sort()

    visualizer_obj = Visualizer()
    return visualizer_obj.mask_to_polygons(ground_truth_mask_root_path, ground_truth_mask_names)


def run_visualize_images(load_path, model_type, img_root_path, ground_truth_mask_root_path):

    ground_truth_mask_names = [file for file in os.listdir(ground_truth_mask_root_path) if file.endswith(".png")]
    ground_truth_mask_names.sort()

    predicted_detection_boxes = np.load(f"{load_path}/{model_type}_boxes.npy", allow_pickle=True)
    ground_truth_boxes = np.load(f"{load_path}/ground_truth_boxes.npy", allow_pickle=True)
    visualizer_obj = Visualizer()

    # -------------------- Boxed Images Visualization -----------------------------
    save_base_path = f"{load_path}/{model_type}_boxes/"
    if os.path.isdir(save_base_path) is False:
        os.mkdir(save_base_path)
    print('save_base_path', save_base_path)

    visualizer_obj.overlay_boxes_over_images(save_base_path, img_root_path, ground_truth_mask_names, ground_truth_boxes,
                                    predicted_detection_boxes, model_type=model_type)
    # bounding_box_per_image_distribution(save_base_path, ground_truth_mask_names, ground_truth_boxes, faster_rcnn_boxes, model_type=model_type)

    # -------------------- Polygon Visualization -----------------------------
    # save_base_path = f"{load_path}/{model_type}_polygon/"
    # if os.path.isdir(save_base_path) is False:
    #     os.mkdir(save_base_path)
    # print('save_base_path', save_base_path)
    #
    # visualizer_obj.overlay_polygons_over_images(save_base_path, img_root_path, ground_truth_mask_names, ground_truth_boxes,
    #                                 predicted_detection_boxes, model_type=model_type)


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
    base_path = 'C:/Users/JunbongJang/PycharmProjects/MARS-Net/'
    ground_truth_mask_root_path = base_path + 'assets/FNA/FNA_test/mask_processed/'
    img_root_path = base_path + 'assets/FNA/FNA_test/img/'
    if 'faster' in model_type:
        # img_root_path = base_path + 'FNA/assets/all-patients/img/'
        # img_root_path = base_path + 'tensorflowAPI/research/object_detection/dataset_tools/assets/images_test/'
        load_path = 'C:/Users/JunbongJang/PycharmProjects/FNA/generated/'

    else:
        # ground_truth_mask_root_path = base_path + 'assets/FNA/FNA_valid_fold0/mask_processed/'
        # img_root_path = base_path + 'assets/FNA/FNA_valid_fold0/img/'
        # load_path = base_path + 'models/results/predict_wholeframe_round1_FNA_VGG19_MTL_auto_reg_aut_input256/FNA_valid_fold0/frame2_A_repeat0/'
        load_path = base_path + 'models/results/predict_wholeframe_round1_FNA_VGG19_MTL_auto_reg_aut_input256_patience_10/FNA_test/frame2_training_repeat0/'

    save_base_path = f'{load_path}/bootstrapped_{model_type}/'
    if os.path.isdir(save_base_path) is False:
        os.mkdir(save_base_path)

    return ground_truth_mask_root_path, img_root_path, load_path, save_base_path


def run_eval(model_type, ground_truth_mask_root_path, img_root_path, load_path, save_base_path):
    test_image_names = get_files_in_folder(img_root_path)
    test_image_names.sort()

    np_predicted_detection_boxes = np.load(f"{load_path}/{model_type}_boxes.npy", allow_pickle=True)
    np_ground_truth_boxes = np.load(f"{load_path}/ground_truth_boxes.npy", allow_pickle=True)
    # bootstrap_data(test_image_names, model_type, 10000, np_predicted_detection_boxes, np_ground_truth_boxes, save_base_path, img_root_path)

    ground_truth_min_follicular = 15
    bootstrapped_box_counts_df = pd.read_csv(f'{save_base_path}bootstrapped_box_counts_df.csv', header=None)
    bootstrap_analysis(bootstrapped_box_counts_df, test_image_names, ground_truth_min_follicular, save_base_path)

    # bootstrapped_area_df = pd.read_csv(f'{save_base_path}bootstrapped_area_df.csv', header=None)
    # ground_truth_mean_area = int(bootstrapped_area_df[0].mean())
    # bootstrap_analysis(bootstrapped_area_df, test_image_names, ground_truth_mean_area, save_base_path)

    # run_visualize_images(load_path, model_type, img_root_path, ground_truth_mask_root_path)
    # run_cam(model_type, img_root_path)


def run_eval_final():
    # get MTL boxes
    model_type = 'MTL_auto_reg_aut'
    ground_truth_mask_root_path, img_root_path, load_path, save_base_path = get_data_path(model_type)
    ground_truth_box_load_path = 'C:/Users/JunbongJang/PycharmProjects/FNA/generated/'
    # ground_truth_boxes = np.load(f"{ground_truth_box_load_path}/ground_truth_boxes.npy", allow_pickle=True)
    mtl_prediction_images_boxes = np.load(f"{load_path}/{model_type}_boxes.npy", allow_pickle=True)
    list_of_ground_truth_polygons = convert_mask_to_polygon(ground_truth_mask_root_path)

    # get Faster R-CNN boxes
    model_type = 'faster_640'
    ground_truth_mask_root_path, img_root_path, load_path, save_base_path = get_data_path(model_type)
    faster_rcnn_prediction_images_boxes = np.load(f"{load_path}/{model_type}_boxes.npy", allow_pickle=True)

    test_image_names = get_files_in_folder(img_root_path)
    test_image_names.sort()
    test_images_by_subject = get_images_by_subject(test_image_names)
    print_images_by_subject_statistics(test_images_by_subject)

    # bootstrap data and count Follicular Clusters in each bootstrap sample
    img_path = os.path.join(img_root_path, test_image_names[0])
    img = Image.open(img_path)  # load images from paths
    img_size = {}
    img_size['height'], img_size['width'] = img.size

    ground_truth_mask_names = [file for file in os.listdir(ground_truth_mask_root_path) if file.endswith(".png")]
    ground_truth_mask_names.sort()

    # -------------------- Polygon Visualization -----------------------------
    # model_type = 'MTL_overlap'
    # save_base_path = f"{load_path}/{model_type}_polygon/"
    # visualizer = Visualizer()
    # visualizer.overlay_two_model_overlapped_polygons_over_images(save_base_path, img_root_path, ground_truth_mask_names,
    #                                                              list_of_ground_truth_polygons, mtl_prediction_images_boxes,
    #                                                              faster_rcnn_prediction_images_boxes, model_type)
    # -----------------------------------------------------
    # Bootstrap number of overlapped polygons per image
    model_type = 'bootstrapped_MTL_overlap'
    save_base_path = f"{load_path}/{model_type}_polygon/"

    bootstrap_two_model_polygons(save_base_path, img_root_path, test_image_names, ground_truth_mask_names, test_images_by_subject, model_type,
                                 list_of_ground_truth_polygons, mtl_prediction_images_boxes, faster_rcnn_prediction_images_boxes, 10000)

    ground_truth_min_follicular = 15
    bootstrapped_df = pd.read_csv(f'{save_base_path}bootstrapped_df.csv', header=None)
    bootstrap_analysis(bootstrapped_df, test_image_names, ground_truth_min_follicular, save_base_path)


if __name__ == "__main__":
    # model_type = 'faster_640'
    # model_type = 'MTL_auto_reg_aut'
    # ground_truth_mask_root_path, img_root_path, load_path, save_base_path = get_data_path(model_type)
    # run_eval(model_type, ground_truth_mask_root_path, img_root_path, load_path, save_base_path)

    run_eval_final()

