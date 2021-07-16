import gc
import argparse
import numpy as np
import time
from vgg_sim_stained import VGG_16
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os.path
from sklearn.utils import shuffle
from tensorflow.keras import backend as K
import tensorflow as tf

dataset = ['all-patients-stained']
def get_args(): 
    parser = argparse.ArgumentParser() 

    parser.add_argument("--input_size", type = int, default = 128)
    parser.add_argument("--cropped_boundary", type = int, default = 30)
    parser.add_argument("--fixed_layers", type = int, default = 18)
    parser.add_argument("--training_dataset", type = str, default = '../crop/cropped_data/')
    parser.add_argument("--batch_size", type = int, default = 4)
    parser.add_argument("--epochs", type = int, default = 100)
    parser.add_argument("--validation_split", type = int, default = 0.2)

    args = parser.parse_args()
    return args

def train_model(num, index, input_size, cropped_boundary, training_dataset, fixed_layers, epochs, batch_size, validation_split):
    if os.path.isfile('average_hist/acc/valid_acc_' + dataset[index] +'.npy'):
        return 
    print(dataset[index])
    
    train_data = np.load(training_dataset + dataset[index] + '_train_mask.npz'); # np.load(training_dataset + dataset[index] + '_' + str(iter) + '_train_mask.npz')
    train_img = train_data['arr_0']
    train_mask = train_data['arr_1']
    
    validation_data = np.load(training_dataset + dataset[index] + '_valid_mask.npz');
    valid_img = validation_data['arr_0']
    valid_mask = validation_data['arr_1']
    
    print('train_img: ', train_img.shape)
    print('train_mask: ', train_mask.shape)
    print('valid_img: ', valid_img.shape)
    print('valid_mask: ', valid_mask.shape)
    
    print(time.time())
    print('Compile Model...')
    model = VGG_16(train_img.shape[2], train_img.shape[3], args.cropped_boundary,0,0)
    print(model.summary())
    print('Fit Model...')
    earlyStopping = EarlyStopping(monitor='val_loss', patience = 5, verbose=1, mode='auto')
    
    model_checkpoint = ModelCheckpoint('average_hist/model/vUnet_'+ dataset[index] +'.hdf5', monitor='val_loss', save_best_only=True)
    hist = model.fit(train_img, train_mask, batch_size = batch_size, epochs = epochs, validation_data=(valid_img,valid_mask), verbose=2, shuffle=True,
              callbacks=[model_checkpoint, earlyStopping])
    print(time.time())
    
    np.save('average_hist/history/history_' + dataset[index]+'.npy', hist.history)
    
    K.clear_session()
    return 


def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

if __name__ == "__main__":
    '''
    # https://kobkrit.com/using-allow-growth-memory-option-in-tensorflow-and-keras-dc8c8081bc96
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    K.tensorflow_backend.set_session(sess)  # set this TensorFlow session as the default 
    '''
    if not os.path.exists('average_hist/history'):
            os.makedirs('average_hist/history')
    if not os.path.exists('average_hist/acc'):
            os.makedirs('average_hist/acc')
    if not os.path.exists('average_hist/model'):
            os.makedirs('average_hist/model')
    #print('--------get_available_gpus------')
    #print(get_available_gpus())

    #Defaults parameters
    args = get_args()
    for index in range(0, len(dataset),1):
        for num in range(1):
            #Training frame
            #for frame in range(30,41,5):
            train_model(num, index, args.input_size, args.cropped_boundary, args.training_dataset, args.fixed_layers, args.epochs, args.batch_size, args.validation_split)
            gc.collect()
