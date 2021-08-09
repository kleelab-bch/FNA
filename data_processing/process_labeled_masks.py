
import os
from scipy import ndimage
import cv2
import numpy as np
from PIL import Image


def image_fill_hole(root_dir, mask_folder, mask_filename, processed_mask_folder):
    output_filename = root_dir + processed_mask_folder + mask_filename
    if os.path.isfile(output_filename) is False:
        print(mask_filename)
        mask_image = cv2.imread(root_dir + mask_folder + mask_filename)

        orig_dtype = mask_image.dtype
        # threshold for each channel
        for i in range(mask_image.shape[2]):
            # 255 is ok since there are only three categories with color red, green and blue
            mask_image[:, :, i] = mask_image[:, :, i] == 255

        # fill hole
        mask_image = ndimage.morphology.binary_fill_holes(mask_image, structure=np.ones((5,5,1))).astype(orig_dtype)
        mask_image = mask_image * 255

        # save file
        cv2.imwrite(output_filename, mask_image)


def remove_secretion_labels(root_dir, mask_folder, mask_filename, processed_mask_folder):
    output_filename = root_dir + processed_mask_folder + mask_filename
    if os.path.isfile(output_filename) is False:
        mask_image = cv2.imread(root_dir + mask_folder + mask_filename)
        print(mask_filename, mask_image.shape)

        orig_dtype = mask_image.dtype
        # 255 is ok since there are only three categories with color red, green and blue
        mask_image[:, :, 0] = 0  # blue channel
        mask_image[:, :, 1] = 0  # green channel

        # save file
        cv2.imwrite(output_filename, mask_image)


if __name__ == "__main__":
    for a_folder in ['FNA_valid_fold0', 'FNA_valid_fold1', 'FNA_valid_fold2', 'FNA_valid_fold3', 'FNA_valid_fold4', 'FNA_valid_fold5']:
        root_dir = f"/media/bch_drive/Public/JunbongJang/Segmentation/assets/FNA/{a_folder}/"
        mask_folder = 'mask/'
        processed_mask_folder = 'mask_processed/'

        if os.path.isdir(root_dir + processed_mask_folder) is False:
            os.mkdir(root_dir + processed_mask_folder)

        mask_files = [f for f in os.listdir(root_dir + mask_folder) if os.path.isfile(root_dir + mask_folder + '/' + f) and f.endswith('.png')]
        for mask_filename in mask_files:
            remove_secretion_labels(root_dir, mask_folder, mask_filename, processed_mask_folder)
            # image_fill_hole(root_dir, mask_folder, mask_filename, save_path)

