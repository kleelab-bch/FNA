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


def run_bootstrap(object_detection_type, img_root_path):
    bootstrap_repetition_num = 10000
    ground_truth_min_follicular = 15
    predicted_min_follicular = 15

    test_image_names = get_files_in_folder(img_root_path)
    np_ground_truth_boxes = np.load("generated/ground_truth_boxes.npy", allow_pickle=True)
    np_faster_rcnn_boxes = np.load("generated/{}_boxes.npy".format(object_detection_type), allow_pickle=True)

    print('ground truth: ', count_boxes(test_image_names, np_ground_truth_boxes))
    print('predictions: ', count_boxes(test_image_names, np_faster_rcnn_boxes))

    # Data Organizing
    test_images_by_subject = get_images_by_subject(test_image_names)
    print_images_by_subject_statistics(test_images_by_subject)

    # Follicular Cluster Counting after bootstrapping
    box_counts = bootstrap.bootstrap_box_counts(test_image_names, test_images_by_subject, np_ground_truth_boxes,
                                      np_faster_rcnn_boxes, bootstrap_repetition_num)
    box_counts_df = pd.DataFrame(box_counts)
    # box_counts_df.to_excel('generated/bootstrapped_box_counts.xlsx', index=False)
    print('boxes shape: ', box_counts.shape)

    # ----------- Analysis Starts -------------------
    save_base_path = 'generated/bootstrapped_{}/'.format(object_detection_type)
    if os.path.isdir(save_base_path) is False:
        os.mkdir(save_base_path)

    # Data Exploration
    bootstrap_viz.plot_histogram(box_counts_df, test_image_names, save_base_path)
    bootstrap_viz.plot_scatter(box_counts_df, ground_truth_min_follicular, save_base_path)

    # Roc curve tutorials
    y_true = box_counts_df[0] >= ground_truth_min_follicular
    # y_pred = box_counts_df[1] >= predicted_min_follicular
    # plot_roc_curve(y_true, y_pred)
    # plot_precision_recall_curve(y_true, y_pred)

    # -------- Varying Predicted Min Follicular Thresholds ------------
    precision_list = []
    recall_list = []
    f1_list = []
    predicted_min_follicular_list = []
    for predicted_min_follicular in range(0, 31):
        a_precision, a_recall, a_f1 = bootstrap.stats_at_threshold(box_counts_df, ground_truth_min_follicular,
                                                         predicted_min_follicular, DEBUG=True)
        predicted_min_follicular_list.append(predicted_min_follicular)
        precision_list.append(a_precision)
        recall_list.append(a_recall)
        f1_list.append(a_f1)

    bootstrap_viz.plot_precision_recall_curve_at_thresholds(y_true, precision_list, recall_list, save_base_path)
    bootstrap_viz.plot_performance_at_thresholds(predicted_min_follicular_list, precision_list, recall_list, f1_list,
                                   save_base_path)


def run_visualize_box_images(load_path, model_type, predict_box_type, img_root_path, ground_truth_mask_root_path):

    ground_truth_mask_names = [file for file in os.listdir(ground_truth_mask_root_path) if file.endswith(".png")]
    ground_truth_mask_names.sort()
    # -------------------- Mask to Box Conversion ----------------------------
    # mask_to_box('generated/', ground_truth_mask_root_path, ground_truth_mask_names, 'ground_truth_3cat')
    # mask_to_box('generated/', vUnet_mask_root_path, ground_truth_mask_names, 'vunet')

    # -------------------- Boxed Images Visualization -----------------------------
    predicted_detection_boxes = np.load(f"{load_path}/{model_type}_boxes.npy", allow_pickle=True)
    # vunet_boxes = np.load("generated/vunet_boxes.npy", allow_pickle=True)
    ground_truth_boxes = np.load(f"{load_path}/ground_truth_boxes.npy", allow_pickle=True)

    save_base_path = f"{load_path}/{model_type}_boxes/"
    if os.path.isdir(save_base_path) is False:
        os.mkdir(save_base_path)
    print('save_base_path', save_base_path)

    visualizer_obj = Visualizer()
    visualizer_obj.overlay_boxes(save_base_path, img_root_path, ground_truth_mask_names, ground_truth_boxes,
                                    predicted_detection_boxes, predict_box_type=predict_box_type)
    # bounding_box_per_image_distribution(save_base_path, ground_truth_mask_names, ground_truth_boxes, faster_rcnn_boxes, predict_box_type=object_detection_type)


def run_cam(object_detection_type, img_root_path):
    predicted_feature_maps = np.load(f"generated/{object_detection_type}_features.npy", allow_pickle=True)
    # feature_name = 'rpn_features_to_crop'  # 40x40x1088
    feature_name= 'rpn_box_predictor_features'  # 40x40x512
    save_heatmap_path = f'generated/{object_detection_type}_boxes/{feature_name}/'
    if os.path.isdir(save_heatmap_path) is False:
        os.mkdir(save_heatmap_path)
    for image_name in tqdm(predicted_feature_maps.item().keys()):
        feature_map = predicted_feature_maps.item()[image_name][feature_name]
        visualizer_obj.visualize_feature_activation_map(feature_map, img_root_path, image_name, save_heatmap_path)


def run_eval_for_tf_objection_detection_API():
    # Data Path and boxes
    base_path = '/media/bch_drive/Public/JunbongJang/'

    ground_truth_mask_root_path = base_path + 'tensorflowAPI/research/object_detection/dataset_tools/assets/masks_test/'
    vUnet_mask_root_path = base_path + 'FNA/vUnet/average_hist/predict_wholeframe/all-patients/all-patients/'

    # img_root_path = base_path + 'FNA/assets/all-patients/img/'
    img_root_path = base_path + 'tensorflowAPI/research/object_detection/dataset_tools/assets/images_test/'
    # img_root_path = base_path + 'tensorflowAPI/research/object_detection/dataset_tools/assets/combined_images_test/'
    object_detection_type = 'faster_640_backup'  # 'faster_640_stained_improved'
    load_path = 'generated'

    # run_bootstrap(object_detection_type, img_root_path)
    run_visualize_box_images(load_path, object_detection_type, 'faster_rcnn', img_root_path, ground_truth_mask_root_path)
    # run_cam(object_detection_type, img_root_path)


def run_eval_for_MTL_classifier():
    # created 7/26/2021
    # Data Path and boxes
    base_path = '/media/bch_drive/Public/JunbongJang/Segmentation/'

    # ground_truth_mask_root_path = base_path + 'assets/FNA/FNA_valid_fold0/mask_processed/'
    # img_root_path = base_path + 'assets/FNA/FNA_valid_fold0/img/'
    # load_path = base_path + 'models/results/predict_wholeframe_round1_FNA_VGG19_MTL_auto_reg_aut_input256/FNA_valid_fold0/frame2_A_repeat0/'

    ground_truth_mask_root_path = base_path + 'assets/FNA/FNA_test/mask_processed/'
    img_root_path = base_path + 'assets/FNA/FNA_test/img/'
    load_path = base_path + 'models/results/predict_wholeframe_round1_FNA_VGG19_MTL_auto_reg_aut_input256_patience_10/FNA_test/frame2_training_repeat0/'

    model_type = 'pred'
    run_visualize_box_images(load_path, model_type, 'MTL_auto_reg_aut', img_root_path, ground_truth_mask_root_path)


if __name__ == "__main__":
    # run_eval_for_tf_objection_detection_API()
    run_eval_for_MTL_classifier()