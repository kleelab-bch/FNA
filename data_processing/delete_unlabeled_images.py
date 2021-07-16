'''
Author Junbong Jang
Modified 6/29/2020 for instance segmentation
'''

import os

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        print(f'The {file_path} does not exist')


def delete_unmatching_file(root_dir, img_folder, mask_folder):
    image_files = [f for f in os.listdir(root_dir + img_folder) if os.path.isfile(root_dir + img_folder + '/' + f) and f.endswith('.png')]
    mask_files = [f for f in os.listdir(root_dir + mask_folder) if os.path.isfile(root_dir + mask_folder + '/' + f) and f.endswith('.png')]

    unlabeled_files = set(image_files) - set(mask_files)
    print(unlabeled_files)
    for unlabeled_file in unlabeled_files:
        delete_file(root_dir + img_folder + '/' + unlabeled_file)


if __name__ == "__main__":
    root_dir = "C:/Users/Junbong/Desktop/FNA Data/all-patients/"
    img_folder = 'images_all'
    mask_folder = 'masks_all'

    delete_unmatching_file(root_dir, img_folder, mask_folder)

