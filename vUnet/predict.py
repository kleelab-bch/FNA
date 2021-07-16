import gc
import argparse
import numpy as np
import time
import cv2
from vgg_sim import VGG_16
from data_generator import prediction_data_generate
import os.path
from sklearn.utils import shuffle
from tensorflow.keras import backend as K

dataset = ['all-patients']

def get_args(): 
    parser = argparse.ArgumentParser() 

    parser.add_argument("--prediciton_path", type = str, default = 'average_hist/predict_wholeframe/')# store the prediction image of the test set.
    parser.add_argument("--img_folder", type = str, default = '/img/')
    parser.add_argument("--mask_folder", type = str, default = '/mask/')
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

def prediction(num, iter, index, saved_path, dataset_folder, img_folder, mask_folder, training_dataset, input_size, output_size):
    print('Prediction...')
    
    # ------------------ get test image indices 
    file1 = open(training_dataset + "crop_info_" + dataset[index] + ".txt","r")  
    file1.readline()
    file1.readline()
    file1.readline()
    file1.readline()
    file1.readline()
    test_image_index_list = file1.readline().strip().replace('imgsTest_index: ', '').split(' ')
    file1.close()
    # ------------------ or use all images as test image
    
    # test_image_index_list = [x for x in range(47)]
   
    #-------------------------------
    # img_path = dataset_folder + 'macrophage_follicular' + img_folder
    # msk_path = dataset_folder + 'macrophage_follicular' + mask_folder
    print(test_image_index_list)
    
    img_path = dataset_folder + dataset[index] + img_folder
    msk_path = dataset_folder + dataset[index] + mask_folder
    #-------------------------------
    train = prediction_data_generate(img_path, msk_path, iter, input_size, output_size, num*10)
    
    #Get the frames for prediction
    imgs_val, namelist, image_cols, image_rows, orig_cols, orig_rows = train.get_whole_frames(test_image_index_list)
    print('img size: ', end='')
    print(image_rows, image_cols)
    print('orig size: ', end='')
    print(orig_rows, orig_cols)
    
    std_mean = np.load(training_dataset + dataset[index] + '_std_mean.npz')
    mean_value = std_mean['arr_0']
    std_value = std_mean['arr_1']
    
    #expand the size of image to fit the model.
    imgs_val = imgs_val[:,np.newaxis,:,:]
    imgs_val = preprocess_input(imgs_val, mean_value, std_value)
    #Build the model.
    model = VGG_16(image_rows,image_cols, 0, image_cols-orig_cols, image_rows-orig_rows)#Size should be adjusted.
    
    #-------------------------------
    model.load_weights('average_hist/model/vUnet'+ '_' + dataset[index] +'.hdf5')  # change it later for self-training
    # only using models trained on one patient images.
    #model.load_weights('average_hist/model/vUnet'+str(iter)+'_'+str(num) + 'bryan' +'.hdf5')		
    #model.load_weights('average_hist/model/vUnet5_0gp2781-first.hdf5')
    #-------------------------------
    
    print(imgs_val.shape)
    pre_m = model.predict(imgs_val, batch_size = 1, verbose = 1)
    pre_m = 255 * pre_m # 0=black color and 255=white color
    print('-----s2_prediction------')
    print(pre_m.shape)
    
    total_number = len(namelist)
    exp = dataset[index]
    write_path = saved_path + exp + "/"
    if os.path.isdir(write_path) == 0:
        os.makedirs(write_path)
    for f in range(total_number):
        print(pre_m[f, 0, :, :].shape)
        out = np.squeeze(pre_m[f, 0, :, :])
        cv2.imwrite(write_path+ namelist[f], out)
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
    #looping datasets
    for index in range(0, len(dataset)):
        prediciton_path_cell = args.prediciton_path + dataset[index] + "/"
        if not os.path.exists(prediciton_path_cell):
            os.makedirs(prediciton_path_cell)
        for num in range(0, 1, 1):
            #Training frame
            #for frame in range(30,41,5):
            prediction(num, 0, index, prediciton_path_cell, args.dataset_folder, args.img_folder, args.mask_folder, args.training_dataset, args.input_size, args.output_size)
            gc.collect()
