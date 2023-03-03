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
from visualizer import Visualizer
from explore_data import get_files_in_folder, get_images_by_subject, print_images_by_subject_statistics
from bootstrapping import bootstrap_analysis, bootstrap_analysis_compare_precision_recall, bootstrap_data, bootstrap_two_model_polygons
import json
import scipy

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


def run_eval_final(root_path, save_path):
    '''
    Here is the brief explanation of evaluation functions called in run_eval_final()

    overlay_two_model_overlapped_polygons_over_images function draws bounding box for every follicular cluster classified to be true in the image patch
    bootstrap_two_model_polygons function bootstraps the samples. This can take several minutes.
    After bootstrapping, bootstrap_analysis function performs analysis such as plotting histogram and precision-recall curves
    bootstrap_analysis_compare_precision_recall can be run at the end after bootstrapping samples from each model (MTL, faster R-CNN, and MTL+faster R-CNN). It draws a precision-recall curve for each model on the same plot for comparison.

    :return:
    '''
    evaluation_dict = {}
    for fold_index, a_letter in zip(range(7), ['A','B','C','D','E','F','G']):
        
        # ground_truth_mask_root_path =  f'{root_path}MARS-Net/assets/FNA/all/mask_processed/'
        # img_root_path = f'{root_path}MARS-Net/assets/FNA/all/img/'

        ground_truth_mask_root_path =  f'{root_path}MARS-Net/assets/FNA/FNA_valid_fold{fold_index}/mask_processed/'
        img_root_path = f'{root_path}MARS-Net/assets/FNA/FNA_valid_fold{fold_index}/img/'
        
        mtl_model_type = 'MTL_auto_reg_aut'  # MTL model with classification and segmentation branches only
        MLT_load_path = f'{root_path}MARS-Net/models/results/predict_wholeframe_round1_FNA_CV_VGG19_{mtl_model_type}_input256_patience_10/'

        faster_rcnn_model_type = 'faster_640'
        faster_rcnn_load_path = f'{root_path}/FNA/assets/{faster_rcnn_model_type}_boxes/'

        mtl_prediction_images_boxes = {}
        faster_rcnn_prediction_images_boxes = {}
        # get MTL boxes
        # mtl_prediction_images_boxes = np.load(f"{MLT_load_path}/{mtl_model_type}_boxes.npy", allow_pickle=True)
        temp_mtl_prediction_image_names = np.load(f"{MLT_load_path}FNA_valid_fold{fold_index}/frame2_{a_letter}_repeat0/image_filenames.npy", allow_pickle=True)
        temp_mtl_prediction_images_boxes = np.load(f"{MLT_load_path}FNA_valid_fold{fold_index}/frame2_{a_letter}_repeat0/pred_boxes.npy", allow_pickle=True)
        temp_mtl_prediction_images_boxes = temp_mtl_prediction_images_boxes.item()

        # get Faster R-CNN boxes
        # faster_rcnn_prediction_images_boxes = np.load(f"{faster_rcnn_load_path}/{faster_rcnn_model_type}_boxes.npy", allow_pickle=True)
        temp_faster_rcnn_prediction_images_boxes = np.load(f"{faster_rcnn_load_path}/{faster_rcnn_model_type}_10{fold_index}_boxes.npy", allow_pickle=True)
        temp_faster_rcnn_prediction_images_boxes = temp_faster_rcnn_prediction_images_boxes.item()

        for mtl_index in range(len(temp_mtl_prediction_images_boxes)):
            mtl_prediction_images_boxes[f"{temp_mtl_prediction_image_names[mtl_index]}.png"] = temp_mtl_prediction_images_boxes[mtl_index]
        
        for image_name, image_boxes in temp_faster_rcnn_prediction_images_boxes.items():
            faster_rcnn_prediction_images_boxes[image_name] = image_boxes

        # remove two names from the faster_rcnn_prediction_images_boxes in the last index
        faster_rcnn_names = list(faster_rcnn_prediction_images_boxes.keys())
        mtl_names = list(mtl_prediction_images_boxes.keys())
        mismatch_names = list(set(faster_rcnn_names) - set(mtl_names))
        if len(mismatch_names) > 0:
            print('mismatch names:', mismatch_names)
            for a_mismatch_name in mismatch_names:
                del faster_rcnn_prediction_images_boxes[a_mismatch_name]
        
        faster_rcnn_names = list(faster_rcnn_prediction_images_boxes.keys())
        mtl_names = list(mtl_prediction_images_boxes.keys())

        assert len(mtl_prediction_images_boxes) == len(faster_rcnn_prediction_images_boxes)
        assert len(mtl_prediction_images_boxes) == len( set(list(faster_rcnn_prediction_images_boxes.keys()) + list(mtl_prediction_images_boxes.keys())) ) 

        # mtl_prediction_images_boxes, faster_rcnn_prediction_images_boxes = aggreage_predicted_boxes_from_cross_validation_by_image_names(MLT_load_path, faster_rcnn_load_path, faster_rcnn_model_type)
        # -----------------------------------------------------------------------------------------------------------

        list_of_ground_truth_polygons = convert_mask_to_polygon(ground_truth_mask_root_path)

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

        # --------------------- Data Loading and preparation above ---------------------------------------

        # model_type options: 'faster-rcnn_overlap', 'MTL_overlap', 'MTL_faster-rcnn_overlap'
        ground_truth_min_follicular = 15  # can be changed to 6 or etc.
        # -------------------- Polygon Visualization -----------------------------
        for model_type in ['MTL_overlap', 'faster-rcnn_overlap', 'MTL_faster-rcnn_overlap']:  # 'MTL_overlap', 'faster-rcnn_overlap', 'MTL_faster-rcnn_overlap'
            save_base_path = f"{save_path}{model_type}_polygon_{fold_index}/"

            visualizer = Visualizer()
            precision, recall, f1, iou = visualizer.overlay_two_model_overlapped_polygons_over_images(save_base_path, img_root_path, ground_truth_mask_names,
                                                                        list_of_ground_truth_polygons, mtl_prediction_images_boxes,
                                                                        faster_rcnn_prediction_images_boxes, model_type)

            evaluation_dict[f'fold{fold_index}_{model_type}'] = {'precision': precision, 'recall': recall, 'f1': f1, 'iou': iou}
            # ------------------- Bootstrapping number of overlapped polygons per image----------------------------------
            # bootstrap_model_type = f'bootstrapped_{model_type}'
            # save_base_path = f"{save_path}{bootstrap_model_type}_polygon/"

            # bootstrap_two_model_polygons(save_base_path, img_root_path, test_image_names, ground_truth_mask_names, test_images_by_subject, bootstrap_model_type,
            #                              list_of_ground_truth_polygons, mtl_prediction_images_boxes, faster_rcnn_prediction_images_boxes, 10000)

            # bootstrapped_df = pd.read_csv(f'{save_base_path}bootstrapped_df.csv', header=None)
            # bootstrap_analysis(bootstrapped_df, test_image_names, ground_truth_min_follicular, save_base_path)

        # ---------- Plot Precision Recall curve for 3 models on bootstrapped samples -----------
        # save_base_path1 = f"{save_path}bootstrapped_MTL_overlap_polygon/"
        # save_base_path2 = f"{save_path}bootstrapped_faster-rcnn_overlap_polygon/"
        # save_base_path3 = f"{save_path}bootstrapped_MTL_faster-rcnn_overlap_polygon/"
        # save_base_path = f"{save_path}bootstrapped_compare_precision_recall_polygon/"
        # if os.path.isdir(save_base_path) is False:
        #     os.mkdir(save_base_path)

        # bootstrapped_df1 = pd.read_csv(f'{save_base_path1}bootstrapped_df.csv', header=None)
        # bootstrapped_df2 = pd.read_csv(f'{save_base_path2}bootstrapped_df.csv', header=None)
        # bootstrapped_df3 = pd.read_csv(f'{save_base_path3}bootstrapped_df.csv', header=None)

        # bootstrap_analysis_compare_precision_recall(bootstrapped_df1, bootstrapped_df2, bootstrapped_df3,
        #                                             ground_truth_min_follicular, save_base_path)

    
    json.dump( evaluation_dict, open( f"{save_path}evaluation_statistics.json", 'w' ) )

    
def aggreage_predicted_boxes_from_cross_validation_by_image_names(MLT_load_path, faster_rcnn_load_path, faster_rcnn_model_type):
    mtl_prediction_images_boxes = {}
    faster_rcnn_prediction_images_boxes = {}
    for fold_index, a_letter in zip(range(7), ['A','B','C','D','E','F','G']):
        # get MTL boxes
        # mtl_prediction_images_boxes = np.load(f"{MLT_load_path}/{mtl_model_type}_boxes.npy", allow_pickle=True)
        temp_mtl_prediction_image_names = np.load(f"{MLT_load_path}FNA_valid_fold{fold_index}/frame2_{a_letter}_repeat0/image_filenames.npy", allow_pickle=True)
        temp_mtl_prediction_images_boxes = np.load(f"{MLT_load_path}FNA_valid_fold{fold_index}/frame2_{a_letter}_repeat0/pred_boxes.npy", allow_pickle=True)
        temp_mtl_prediction_images_boxes = temp_mtl_prediction_images_boxes.item()

        # get Faster R-CNN boxes
        # faster_rcnn_prediction_images_boxes = np.load(f"{faster_rcnn_load_path}/{faster_rcnn_model_type}_boxes.npy", allow_pickle=True)
        temp_faster_rcnn_prediction_images_boxes = np.load(f"{faster_rcnn_load_path}/{faster_rcnn_model_type}_10{fold_index}_boxes.npy", allow_pickle=True)
        temp_faster_rcnn_prediction_images_boxes = temp_faster_rcnn_prediction_images_boxes.item()

        for mtl_index in range(len(temp_mtl_prediction_images_boxes)):
            mtl_prediction_images_boxes[f"{temp_mtl_prediction_image_names[mtl_index]}.png"] = temp_mtl_prediction_images_boxes[mtl_index]
        
        for image_name, image_boxes in temp_faster_rcnn_prediction_images_boxes.items():
            faster_rcnn_prediction_images_boxes[image_name] = image_boxes

        # remove two names from the faster_rcnn_prediction_images_boxes in the last index
        faster_rcnn_names = list(faster_rcnn_prediction_images_boxes.keys())
        mtl_names = list(mtl_prediction_images_boxes.keys())
        mismatch_names = list(set(faster_rcnn_names) - set(mtl_names))
        if len(mismatch_names) > 0:
            print('mismatch names:', mismatch_names)
            for a_mismatch_name in mismatch_names:
                del faster_rcnn_prediction_images_boxes[a_mismatch_name]
        
        
        faster_rcnn_names = list(faster_rcnn_prediction_images_boxes.keys())
        mtl_names = list(mtl_prediction_images_boxes.keys())
        print('faster', len(faster_rcnn_names), 'mtl', len(mtl_names))

        assert len(mtl_prediction_images_boxes) == len(faster_rcnn_prediction_images_boxes)
        assert len(mtl_prediction_images_boxes) == len( set(list(faster_rcnn_prediction_images_boxes.keys()) + list(mtl_prediction_images_boxes.keys())) ) 

    return mtl_prediction_images_boxes, faster_rcnn_prediction_images_boxes

def print_eval_final(save_path):

    evaluation_dict = json.load( open( f"{save_path}evaluation_statistics.json", 'r' ) )

    summary_eval_dict = {}
    print('------------------- Print Statistics -------------------')
    for model_type in ['MTL_overlap', 'faster-rcnn_overlap', 'MTL_faster-rcnn_overlap']:
        print(model_type)
        for metric_name in ['precision', 'recall', 'f1', 'iou']:
            metric_value_list = []
            for a_fold_index in range(7):
                metric_value_list.append(evaluation_dict[f'fold{a_fold_index}_{model_type}'][metric_name])
                
            np_metric_value = np.array(metric_value_list)
            print(metric_name, '     avg:', np.mean(np_metric_value), '   std:', np.std(np_metric_value))

            if model_type not in summary_eval_dict.keys():
                summary_eval_dict[model_type] = {metric_name: np_metric_value }
            else:
                summary_eval_dict[model_type][metric_name] = np_metric_value
        print('--------------')
    
    
    print('------------------- Hypothesis Testings -------------------')
    print('MTL vs. Faster R-CNN')
    perform_wilcoxon_signed_rank_sum_test(summary_eval_dict['MTL_overlap']['precision'], summary_eval_dict['faster-rcnn_overlap']['precision'])
    perform_wilcoxon_signed_rank_sum_test(summary_eval_dict['MTL_overlap']['recall'], summary_eval_dict['faster-rcnn_overlap']['recall'])
    perform_wilcoxon_signed_rank_sum_test(summary_eval_dict['MTL_overlap']['f1'], summary_eval_dict['faster-rcnn_overlap']['f1'])
    perform_wilcoxon_signed_rank_sum_test(summary_eval_dict['MTL_overlap']['iou'], summary_eval_dict['faster-rcnn_overlap']['iou'])

    print('MTL vs. FNA-Net')
    perform_wilcoxon_signed_rank_sum_test(summary_eval_dict['MTL_overlap']['precision'], summary_eval_dict['MTL_faster-rcnn_overlap']['precision'])
    perform_wilcoxon_signed_rank_sum_test(summary_eval_dict['MTL_overlap']['recall'], summary_eval_dict['MTL_faster-rcnn_overlap']['recall'])
    perform_wilcoxon_signed_rank_sum_test(summary_eval_dict['MTL_overlap']['f1'], summary_eval_dict['MTL_faster-rcnn_overlap']['f1'])
    perform_wilcoxon_signed_rank_sum_test(summary_eval_dict['MTL_overlap']['iou'], summary_eval_dict['MTL_faster-rcnn_overlap']['iou'])

    print('Faster R-CNN vs. FNA-Net')
    perform_wilcoxon_signed_rank_sum_test(summary_eval_dict['faster-rcnn_overlap']['precision'], summary_eval_dict['MTL_faster-rcnn_overlap']['precision'])
    perform_wilcoxon_signed_rank_sum_test(summary_eval_dict['faster-rcnn_overlap']['recall'], summary_eval_dict['MTL_faster-rcnn_overlap']['recall'])
    perform_wilcoxon_signed_rank_sum_test(summary_eval_dict['faster-rcnn_overlap']['f1'], summary_eval_dict['MTL_faster-rcnn_overlap']['f1'])
    perform_wilcoxon_signed_rank_sum_test(summary_eval_dict['faster-rcnn_overlap']['iou'], summary_eval_dict['MTL_faster-rcnn_overlap']['iou'])

    Visualizer().manuscript_draw_comparison_bar_graph_with_errors(save_path, summary_eval_dict)


def perform_wilcoxon_signed_rank_sum_test(first_list, second_list):
    # ttest_rel or wilcoxon does not change the p-value much either and accept/reject results do not change
    res = scipy.stats.wilcoxon(first_list, second_list)

    print('p', res.pvalue, '  statistic', res.statistic)


if __name__ == "__main__":
    # root_path = 'C:/Users/JunbongJang/PycharmProjects/'  
    root_path = 'C:/Users/Jun/Documents/PycharmProjects/'
    save_path = f"{root_path}FNA/generated/CV_7folds/"

    # model_type = 'faster_640'
    # model_type = 'MTL_auto_reg_aut'
    # ground_truth_mask_root_path, img_root_path, load_path, save_base_path = get_data_path(model_type)
    # run_eval(model_type, ground_truth_mask_root_path, img_root_path, load_path, save_base_path)

    # run_eval_final(root_path, save_path)
    print_eval_final(save_path)

