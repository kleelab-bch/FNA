import gc
import argparse
import numpy as np
import time
import cv2
from vgg_sim_stained import VGG_16
from data_generator_stained import prediction_data_generate
import os.path
from sklearn.utils import shuffle
from tensorflow.keras import backend as K
from PIL import Image

dataset = ['all-patients-stained']

def get_args(): 
    parser = argparse.ArgumentParser() 

    parser.add_argument("--prediciton_path", type = str, default = 'average_hist/predict_wholeframe/')# store the prediction image of the test set.
    parser.add_argument("--img_folder", type = str, default = '/images_valid/')
    parser.add_argument("--dataset_folder", type = str, default = '../assets/')
    parser.add_argument("--training_dataset", type = str, default = '../crop/cropped_data/')
    parser.add_argument("--input_size", type = int, default = 128)
    parser.add_argument("--output_size", type = int, default = 68)

    args = parser.parse_args()
    return args


def preprocess_input(imgs, mean, std):
    imgs_p = np.repeat(imgs, 3, axis=1)
    imgs_p = imgs_p.astype('float32')
    imgs_p -= mean
    imgs_p /= std
    
    return imgs_p
    

def prediction(index, saved_path, dataset_folder, img_folder, training_dataset, input_size, output_size):
    print('Prediction...')
    
    img_path = dataset_folder + dataset[index] + img_folder
    train = prediction_data_generate(img_path, 13)
    
    #Get the frames for prediction
    images, namelist, image_cols, image_rows, orig_cols, orig_rows = train.get_whole_frames()
    print('img size: ', end='')
    
    crop_cols = image_cols-orig_cols
    crop_rows = image_rows-orig_rows
    image_cols = 1024
    image_rows = 1024
    #expand the size of image to fit the model.
    images = images[:,np.newaxis,:image_rows,:image_cols]
    
    print(images.shape)
    mean_value = np.mean(images)
    std_value = np.std(images)
    images = preprocess_input(images, mean_value, std_value)
    
    #Build the model.
    model = VGG_16(image_rows,image_cols, 0, 0, 0)
    model.load_weights('average_hist/model/vUnet'+ '_' + dataset[index] +'.hdf5')  # change it later for self-training

    print('-----prediction------')
    pre_m = model.predict(images, batch_size = 1, verbose = 2)
    pre_m = 255 * pre_m
    
    print(pre_m.shape)
    
    total_number = len(namelist)
    exp = dataset[index]
    write_path = saved_path + exp + "/"
    if os.path.isdir(write_path) == 0:
        os.makedirs(write_path)
    for image_index in range(total_number):
        out = pre_m[image_index,:,:,:]
        out = np.swapaxes(out,0,2)
        out = np.swapaxes(out,0,1)
        print(out.shape)
        cv2.imwrite(write_path+ "predict" + namelist[image_index], out)
    K.clear_session()
    


if __name__ == "__main__":
    if not os.path.exists('average_hist/history'):
            os.makedirs('average_hist/history')
    if not os.path.exists('average_hist/acc'):
            os.makedirs('average_hist/acc')
    if not os.path.exists('average_hist/model'):
            os.makedirs('average_hist/model')

    #Defaults parameters
    args = get_args()
    
    for index in range(0, len(dataset)):
        prediciton_path_cell = args.prediciton_path + dataset[index] + "/"
        if not os.path.exists(prediciton_path_cell):
            os.makedirs(prediciton_path_cell)
        prediction( index, prediciton_path_cell, args.dataset_folder, args.img_folder, args.training_dataset, args.input_size, args.output_size)
        gc.collect()
