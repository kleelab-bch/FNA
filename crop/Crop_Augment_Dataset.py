import gc
import argparse
import numpy as np
import time
from train_val_generate_stained import data_generate, augment_data, preprocess_input, preprocess_output
import os.path
from tensorflow.keras import backend as K
import datetime
import psutil

dataset = ['all-patients-stained']
# dataset = ['gp2781-first', 'gp2781-fourth', 'gp2781-third', 'ha2779-first', 'ha2779-third']


def get_args(): 
    parser = argparse.ArgumentParser() 

    parser.add_argument("--input_size", type = int, default = 128)
    parser.add_argument("--output_size", type = int, default = 68)
    parser.add_argument("--randomseed", type = int, default = 0)
    parser.add_argument("--augmentation_factor", type = int, default = 50)
    parser.add_argument("--saved_folder", type = str, default = './cropped_data/')
    parser.add_argument("--img_folder", type = str, default = '/img/')
    parser.add_argument("--mask_folder", type = str, default = '/mask/')
    parser.add_argument("--dataset_folder", type = str, default = '../assets/')
    parser.add_argument("--img_format", type = str, default = '.png')
    parser.add_argument("--crop_patches", type = int, default = 1000)

    args = parser.parse_args()
    return args

def Training_dataset(num, index, saved_folder, input_size, output_size, randomseed, augmentation_factor, img_folder, mask_folder, dataset_folder, img_format, cropped_patches):
    print(dataset[index], num)
    print(datetime.datetime.now())
    
    #train_image_path = '../../TfResearch/object_detection/dataset_tools/assets/images_train/'
    #valid_image_path = '../../TfResearch/object_detection/dataset_tools/assets/images_valid/'
    #test_image_path = '../../TfResearch/object_detection/dataset_tools/assets/images_test/'
    root_image_path = '../assets/' + dataset[index] + '/'
    crop_type = 'train'
    
    train = data_generate(dataset[index], input_size, output_size, randomseed, saved_folder, img_format, cropped_patches, dataset_folder, img_folder, mask_folder)
    imgs_train, msks_train, edgs_train, mean_value, std_value, imgsTrain_index, imgsVal_index, imgsTest_index = train.crop(saved_folder + "crop_info_"+dataset[index]+".txt", root_image_path, crop_type)
    np.save(saved_folder + dataset[index] + '_' + crop_type + '_frames.npy', imgsTrain_index)
    np.savez(saved_folder + dataset[index] + '_' + crop_type + '_std_mean.npz', mean_value, std_value)
    
    print(imgs_train.shape)
    imgs_train = np.swapaxes(imgs_train,1,3)
    imgs_train = np.swapaxes(imgs_train,2,3)
    msks_train = msks_train[:,np.newaxis,:,:]
    edgs_train = edgs_train[:,np.newaxis,:,:]
    
    print(imgs_train.shape)
    print(msks_train.shape)
    
    # imgs_train, msks_train, edgs_train = augment_data(imgs_train, msks_train, edgs_train, augmentation_factor)
    train = preprocess_input(imgs_train, input_size, input_size, mean_value, std_value)
    mask = preprocess_output(msks_train, output_size, output_size)
    edge = preprocess_output(edgs_train, output_size, output_size)

    print('Saving...')
    np.savez(saved_folder + dataset[index] + '_' + crop_type + '_mask.npz', train, mask, edge)
    
    K.clear_session()
    print(datetime.datetime.now())
    return 


if __name__ == "__main__":
    #Defaults parameters
    args = get_args()
    if not os.path.exists(args.saved_folder):
        os.makedirs(args.saved_folder)
    for index in range(0, len(dataset),1):
        for num in range(1):
            Training_dataset(num, index, args.saved_folder, args.input_size, args.output_size, args.randomseed, args.augmentation_factor, args.img_folder, args.mask_folder, args.dataset_folder, args.img_format, args.crop_patches)
            gc.collect()

