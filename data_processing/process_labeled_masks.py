
import os
from scipy import ndimage
import cv2
import numpy as np
from PIL import Image

def image_fill_hole(root_dir, mask_folder, output_filename):
    if os.path.isfile(output_filename) is False:
        print(mask_file)
        mask_image = cv2.imread(root_dir + mask_folder + '/' + mask_file)

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


if __name__ == "__main__":
    root_dir = "C:/Users/Junbong/Desktop/FNA Data/all-patients/"
    processed_dir = root_dir + 'processed_masks/'

    if os.path.isdir(processed_dir) is False:
        os.mkdir(processed_dir)

    img_folder = 'images_all'
    mask_folder = 'masks_all'

    mask_files = [f for f in os.listdir(root_dir + mask_folder) if os.path.isfile(root_dir + mask_folder + '/' + f) and f.endswith('.png')]
    for mask_file in mask_files:
        output_filename = processed_dir + mask_file
        image_fill_hole(root_dir, mask_folder, output_filename)

