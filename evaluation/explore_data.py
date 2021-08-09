"""
Author Junbong Jang
Date: 7/30/2020

For exploration of the FNA data
"""

import os


def get_images_by_subject(image_names):
    # find unique subject id from list of image names
    unique_subjects = set()
    images_by_subject = {}
    for image_name in image_names:
        unique_subjects.add(image_name[:6])  # duplicate name will be ignored in the set
        if image_name[:6] not in images_by_subject:
            images_by_subject[image_name[:6]] = []
        images_by_subject[image_name[:6]].append(image_name)

    return images_by_subject


def print_images_by_subject_statistics(images_by_subject):
    total_images = 0
    for subject in images_by_subject.keys():
        image_num = len(images_by_subject[subject])
        total_images = total_images + image_num

    for subject in images_by_subject.keys():
        image_num = len(images_by_subject[subject])
        print(subject, image_num, round(image_num/total_images,3))

    print('--------------')


def get_files_in_folder(folder_path):
    return [f for f in os.listdir(folder_path) if os.path.isfile(folder_path + '/' + f) and f.endswith('.png')]


def check_img_mask_matches():
    root_dir = "C:/Users/Junbong/Desktop/FNA Data/all-patients/"
    for img_type in ['images', 'masks']:
        for split_type in ['train', 'valid', 'test']:
            print(img_type, split_type)
            folder_path = root_dir + img_type + '_' + split_type
            files = get_files_in_folder(folder_path)
            print(len(files))


if __name__ == "__main__":
    train_image_names = get_files_in_folder("C:/Users/Junbong/Desktop/FNA Data/all-patients/images_train")
    valid_image_names = get_files_in_folder("C:/Users/Junbong/Desktop/FNA Data/all-patients/images_valid")
    test_image_names = get_files_in_folder("C:/Users/Junbong/Desktop/FNA Data/all-patients/images_test")

    images_by_subject = get_images_by_subject(train_image_names + valid_image_names + test_image_names)
    print_images_by_subject_statistics(images_by_subject)

    # images_by_subject = get_images_by_subject(train_image_names)
    # images_by_subject = get_images_by_subject(valid_image_names)
    # images_by_subject = get_images_by_subject(test_image_names)