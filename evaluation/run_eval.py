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

from calc_polygon import connect_overlapped_boxes, count_overlap_polygons
from visualizer import Visualizer
from explore_data import get_files_in_folder, get_images_by_subject, print_images_by_subject_statistics
from bootstrapping import bootstrap_analysis, bootstrap_data


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
    base_path = 'C:/Users/Jun/Documents/PycharmProjects/MARS-Net/'
    ground_truth_mask_root_path = base_path + 'assets/FNA/FNA_test/mask_processed/'
    img_root_path = base_path + 'assets/FNA/FNA_test/img/'
    if 'faster' in model_type:
        # img_root_path = base_path + 'FNA/assets/all-patients/img/'
        # img_root_path = base_path + 'tensorflowAPI/research/object_detection/dataset_tools/assets/images_test/'
        load_path = 'C:/Users/Jun/Documents/PycharmProjects/FNA/generated/'

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
    ground_truth_box_load_path = 'C:/Users/Jun/Documents/PycharmProjects/FNA/generated/'
    ground_truth_boxes = np.load(f"{ground_truth_box_load_path}/ground_truth_boxes.npy", allow_pickle=True)
    mtl_prediction_images_boxes = np.load(f"{load_path}/{model_type}_boxes.npy", allow_pickle=True)

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
    model_type = 'MTL_faster-rcnn_overlap'
    save_base_path = f"{load_path}/{model_type}_polygon/"
    if os.path.isdir(save_base_path) is False:
        os.mkdir(save_base_path)
    print('save_base_path', save_base_path)

    total_false_negative = 0
    total_false_positive = 0
    total_gt_overlaps = 0

    visualizer = Visualizer()
    for idx, filename in enumerate(ground_truth_mask_names):
        img_path = os.path.join(img_root_path, filename)
        img = Image.open(img_path)  # load images from paths

        orig_filename = filename.replace('predict', '').replace('stained_', '')
        one_ground_truth_boxes = ground_truth_boxes.item()[orig_filename]
        faster_rcnn_predicted_boxes = faster_rcnn_prediction_images_boxes.item()[filename]

        faster_rcnn_predicted_boxes[:, 0], faster_rcnn_predicted_boxes[:, 2] = faster_rcnn_predicted_boxes[:,
                                                               0] * img.height, faster_rcnn_predicted_boxes[:,
                                                                                2] * img.height  # 1944
        faster_rcnn_predicted_boxes[:, 1], faster_rcnn_predicted_boxes[:, 3] = faster_rcnn_predicted_boxes[:,
                                                               1] * img.width, faster_rcnn_predicted_boxes[:,
                                                                               3] * img.width  # 2592

        mtl_predicted_boxes = mtl_prediction_images_boxes.item()[idx]

        if one_ground_truth_boxes.shape[0] > 0 or mtl_predicted_boxes.shape[0] > 0 or faster_rcnn_predicted_boxes.shape[0] > 0:
            # if there is at least one prediction or ground truth box in the image

            # convert MTL boxes to polygons
            # convert Faster R-CNN boxes to polygons
            # convert ground truth boxes to polygons
            faster_rcnn_polygon = connect_overlapped_boxes(faster_rcnn_predicted_boxes)
            mtl_polygon = connect_overlapped_boxes(mtl_predicted_boxes)
            ground_truth_polygon = connect_overlapped_boxes(one_ground_truth_boxes)

            # overlap MTL polygons with Faster R-CNN polygons
            overlapped_prediction_polygon = faster_rcnn_polygon.intersection(mtl_polygon)
            # overlap MTL + Faster R-CNN polygon with ground truth polygon
            overlapped_final_polygon = overlapped_prediction_polygon.intersection(ground_truth_polygon)
            # Count overlapped polygons
            gt_overlaps, false_negative, false_positive, gt_overlap_pair = count_overlap_polygons(ground_truth_polygon, overlapped_prediction_polygon)
            if gt_overlaps > 0 or false_negative > 0 or false_positive > 0:
                print(idx, filename)
                print(gt_overlaps, false_negative, false_positive)
            total_gt_overlaps = total_gt_overlaps + gt_overlaps
            total_false_negative = total_false_negative + false_negative
            total_false_positive = total_false_positive + false_positive

            # overlay polygon over the image
            overlaid_img = visualizer.overlay_polygons(img, ground_truth_polygon, (255, 0, 0), True)
            overlaid_img = visualizer.overlay_polygons(overlaid_img, overlapped_prediction_polygon, (0, 255, 0), True)
            overlaid_img = visualizer.overlay_polygons(overlaid_img, overlapped_final_polygon, (255, 255, 0), True)
            save_filename = save_base_path + filename
            overlaid_img.save(save_filename)

    # evaluate by F1, Precision and Recall
    print('tp:', total_gt_overlaps, 'fn:', total_false_negative, 'fp:', total_false_positive)
    precision = total_gt_overlaps / (total_gt_overlaps + total_false_positive)
    recall = total_gt_overlaps / (total_gt_overlaps + total_false_negative)
    f1 = 2*precision*recall/(precision+recall)
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)

    # -----------------------------------------------------
    # Bootstrap number of overlapped polygons per image
    # model_type = 'bootstrapped_MTL_faster-rcnn_polygon'
    # save_base_path = f"{load_path}/{model_type}_polygon/"
    # if os.path.isdir(save_base_path) is False:
    #     os.mkdir(save_base_path)
    # print('save_base_path', save_base_path)
    #
    # image_name_to_index = {}
    # for i, image_name in enumerate(test_image_names):
    #     image_name_to_index[image_name] = i
    #
    # from sklearn.utils import resample
    # bootstrap_repetition_num = 10000
    # testset_sample_size = len(ground_truth_mask_names)
    # box_counts = np.zeros(shape=(bootstrap_repetition_num, 2))
    #
    # for bootstrap_repetition_index in tqdm(range(bootstrap_repetition_num)):
    #     images_by_subject = test_images_by_subject
    #     # ------- bootstrap subjects ------
    #     if images_by_subject != None:
    #         bootstrap_sampled_subjects = resample(list(images_by_subject.keys()), replace=True, n_samples=len(images_by_subject.keys()),
    #                                              random_state=bootstrap_repetition_index)
    #         # only get images from sampled subjects
    #         image_names = []
    #         for bootstrap_sampled_subject in bootstrap_sampled_subjects:
    #             image_names = image_names + images_by_subject[bootstrap_sampled_subject]
    #
    #     # ------- bootstrap images ---------
    #     bootstrap_sampled_image_names = resample(image_names, replace=True, n_samples=testset_sample_size,
    #                                          random_state=bootstrap_repetition_index)
    #
    #     ground_truth_boxes_total = 0
    #     prediction_boxes_total = 0
    #
    #     for idx, filename in enumerate(bootstrap_sampled_image_names):
    #         # faster_rcnn_prediction_images_boxes = np.load(f"{load_path}/faster_640_boxes.npy", allow_pickle=True)
    #         img_path = os.path.join(img_root_path, filename)
    #         img = Image.open(img_path)  # load images from paths
    #
    #         orig_filename = filename.replace('predict', '').replace('stained_', '')
    #         one_ground_truth_boxes = ground_truth_boxes.item()[orig_filename]
    #         faster_rcnn_predicted_boxes = faster_rcnn_prediction_images_boxes.item()[filename].copy()
    #
    #         faster_rcnn_predicted_boxes[:, 0], faster_rcnn_predicted_boxes[:, 2] = faster_rcnn_predicted_boxes[:,
    #                                                                0] * img.height, faster_rcnn_predicted_boxes[:,
    #                                                                                 2] * img.height  # 1944
    #         faster_rcnn_predicted_boxes[:, 1], faster_rcnn_predicted_boxes[:, 3] = faster_rcnn_predicted_boxes[:,
    #                                                                1] * img.width, faster_rcnn_predicted_boxes[:,
    #                                                                                3] * img.width  # 2592
    #         mtl_predicted_boxes = mtl_prediction_images_boxes.item()[image_name_to_index[filename]]
    #
    #         if one_ground_truth_boxes.shape[0] > 0 or mtl_predicted_boxes.shape[0] > 0 or faster_rcnn_predicted_boxes.shape[0] > 0:
    #             # if there is at least one prediction or ground truth box in the image
    #
    #             # convert MTL boxes to polygons
    #             # convert Faster R-CNN boxes to polygons
    #             # convert ground truth boxes to polygons
    #             faster_rcnn_polygon = connect_overlapped_boxes(faster_rcnn_predicted_boxes)
    #             mtl_polygon = connect_overlapped_boxes(mtl_predicted_boxes)
    #             ground_truth_polygon = connect_overlapped_boxes(one_ground_truth_boxes)
    #
    #             # overlap MTL polygons with Faster R-CNN polygons
    #             # overlapped_prediction_polygon = faster_rcnn_polygon.intersection(mtl_polygon)
    #             overlapped_prediction_polygon = faster_rcnn_polygon
    #
    #             # preprocess before counting to prevent error
    #             if ground_truth_polygon is None:
    #                 ground_truth_polygon = []
    #             elif ground_truth_polygon.geom_type == 'Polygon':
    #                 ground_truth_polygon = [ground_truth_polygon]
    #             if overlapped_prediction_polygon is None:
    #                 overlapped_prediction_polygon = []
    #             elif overlapped_prediction_polygon.geom_type == 'Polygon':
    #                 overlapped_prediction_polygon = [overlapped_prediction_polygon]
    #
    #             ground_truth_boxes_total = ground_truth_boxes_total + len(ground_truth_polygon)
    #             prediction_boxes_total = prediction_boxes_total + len(overlapped_prediction_polygon)
    #             box_counts[bootstrap_repetition_index, :] = ground_truth_boxes_total, prediction_boxes_total
    #
    # box_counts_df = pd.DataFrame(box_counts)
    # box_counts_df.to_csv(f'{save_base_path}bootstrapped_df.csv', index=False, header=False)
    #
    # ground_truth_min_follicular = 15
    # bootstrapped_df = pd.read_csv(f'{save_base_path}bootstrapped_df.csv', header=None)
    # bootstrap_analysis(bootstrapped_df, test_image_names, ground_truth_min_follicular, save_base_path)


if __name__ == "__main__":
    # model_type = 'faster_640'  # 'faster_640_stained_improved'
    # model_type = 'MTL_auto_reg_aut'

    # ground_truth_mask_root_path, img_root_path, load_path, save_base_path = get_data_path(model_type)
    # run_eval(model_type, ground_truth_mask_root_path, img_root_path, load_path, save_base_path)

    run_eval_final()