"""
Author Junbong Jang
Modified 7/8/2020

To split images into train and validation set for the FNA project
"""

import random
import os
from PIL import Image
import numpy as np
import shutil
from tqdm import tqdm


def split_image_by_category(mask_folder_path, mask_file_names, follicular_test_filenames):
    # split images into follicular (red, intensity value 76) and secretion
    secretion_file_names = []
    follicular_file_names = []

    for mask_file_name in tqdm(mask_file_names):
        # print(mask_file_name, end='  ')
        mask_file_path = mask_folder_path + mask_file_name
        a_mask = Image.open(mask_file_path)
        mask_np = np.asarray(a_mask.convert('L'))

        # print(np.unique(mask_np), end='  ')
        pixel_val = 76
        nonbackground_indices_x = np.any(mask_np == pixel_val, axis=0)
        nonbackground_indices_y = np.any(mask_np == pixel_val, axis=1)
        nonzero_x_indices = np.where(nonbackground_indices_x)
        nonzero_y_indices = np.where(nonbackground_indices_y)

        if np.asarray(nonzero_x_indices).shape[1] == 0 and np.asarray(nonzero_y_indices).shape[1] == 0:
            secretion_file_names.append(mask_file_name)
            # print('secretion')
        else:
            if mask_file_name not in follicular_test_filenames:
                follicular_file_names.append(mask_file_name)
                # print('follicular')
            else:
                print('skipped')

    print('secretion_file_names length:', len(secretion_file_names))
    print('follicular_file_names length:', len(follicular_file_names))

    return secretion_file_names, follicular_file_names


def move_files_to_another_folder(root_dir, split_type, secretion_image_list, follicular_image_list):
    image_destination_path = root_dir + split_type + '/img/'
    mask_destination_path = root_dir + split_type + '/mask/'
    if os.path.isdir(image_destination_path) is False:
        os.mkdir(image_destination_path)
    if os.path.isdir(mask_destination_path) is False:
        os.mkdir(mask_destination_path)

    for image_name in tqdm(secretion_image_list+follicular_image_list):
        image_source_path = root_dir + 'FNA_train_val/img/' + image_name
        mask_source_path = root_dir + 'FNA_train_val/mask/' + image_name

        shutil.copyfile(image_source_path, image_destination_path + image_name)
        shutil.copyfile(mask_source_path, mask_destination_path + image_name)


if __name__ == "__main__":
    random.seed(a=42, version=2)

    root_dir = "../../Segmentation/assets/FNA/"

    follicular_test_filenames = []
    # follicular_test_filenames = ['ha2780-third-0331.png', 'ha2780-third-0275.png', 'gp2781-first-0410.png', 'gp2781-first-0477.png',
    #                              'gp2781-third-0578.png', 'gp2781-fourth-0581.png', 'jm2776-first-0142.png', 'jm2775-third-0236.png']
    orig_mask_folder_path = root_dir + 'FNA_train_val/mask/'

    # ------------------- User Parameter Setting Done --------------------------
    mask_file_names = [f for f in os.listdir(orig_mask_folder_path) if os.path.isfile(orig_mask_folder_path + f) and f.endswith('.png')]
    print('mask_file_names length', len(mask_file_names))

    secretion_file_names, follicular_file_names = split_image_by_category(orig_mask_folder_path, mask_file_names, follicular_test_filenames)

    # shuffle lists
    random.shuffle(secretion_file_names)
    random.shuffle(follicular_file_names)

    # --------- sampling from each category for train/val/test split ---------
    secretion_sample_num = 35
    follicular_sample_num = 8
    secretion_index_list = random.sample(range(0, len(secretion_file_names)-1), secretion_sample_num)
    secretion_image_list = [secretion_file_names[i] for i in secretion_index_list]
    follicular_index_list = random.sample(range(0, len(follicular_file_names)-1), follicular_sample_num)
    follicular_image_list = [follicular_file_names[i] for i in follicular_index_list]
    split_type = 'test'  # valid or test
    # if split_type == 'test':
    #     follicular_image_list = follicular_test_filenames

    print('sampled secretion:', secretion_image_list)
    print('sampled follicular:', follicular_image_list)

    move_files_to_another_folder(root_dir, split_type)

    # --------- N-fold cross validation --------------
    # total_fold = 6
    #
    # for a_fold in range(total_fold):
    #     print('a_fold', a_fold)
    #
    #     assert ((len(secretion_file_names)/total_fold - int(len(secretion_file_names)/total_fold)) == 0)
    #     secretion_low_index = a_fold*len(secretion_file_names)//total_fold
    #     secretion_high_index = (a_fold+1)*len(secretion_file_names)//total_fold
    #     follicular_low_index = a_fold*len(follicular_file_names)//total_fold
    #     follicular_high_index = (a_fold+1)*len(follicular_file_names)//total_fold
    #     print(secretion_low_index, secretion_high_index)
    #     print(follicular_low_index, follicular_high_index)
    #
    #     val_secretion_file_names = secretion_file_names[secretion_low_index:secretion_high_index]
    #     val_follicular_file_names = follicular_file_names[follicular_low_index:follicular_high_index]
    #
    #     train_secretion_file_names = [item for item in secretion_file_names if item not in val_secretion_file_names]
    #     train_follicular_file_names = [item for item in follicular_file_names if item not in val_follicular_file_names]
    #
    #     print('val_secretion_file_names', len(val_secretion_file_names))
    #     print('val_follicular_file_names', len(val_follicular_file_names))
    #     print('train_secretion_file_names', len(train_secretion_file_names))
    #     print('train_follicular_file_names', len(train_follicular_file_names))
    #
    #     move_files_to_another_folder(root_dir, f'FNA_valid_fold{a_fold}', val_secretion_file_names, val_follicular_file_names)
    #     move_files_to_another_folder(root_dir, f'FNA_train_fold{a_fold}', train_secretion_file_names, train_follicular_file_names)

