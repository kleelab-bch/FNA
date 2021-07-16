from __future__ import print_function
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, Conv2DTranspose,BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#K.set_image_dim_ordering('th')  # Theano dimension ordering in this code
K.set_image_data_format('channels_first')
smooth = 1.


def dice_coef(y_true, y_pred):
    #true_thresh = tf.cast(y_true > 127, tf.float32)
    #pred_thresh = tf.cast(y_pred > 127, tf.float32)

    y_true_f = K.flatten(K.round(y_true))
    y_pred_f = K.flatten(K.round(y_pred))
    
    intersection = K.sum(y_true_f * y_pred_f)
    
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))


def weighted_cross_entropy(beta):
  def convert_to_logits(y_pred):
      # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
      y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
      # logistic function but inverse
      # map probability from 0 to 1 ----> Real Number from negative infinity to positive infinity
      return tf.log(y_pred / (1 - y_pred))

  def loss(y_true, y_pred):
    y_pred = convert_to_logits(y_pred)
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=beta)

    # or reduce_sum and/or axis=-1
    return tf.reduce_mean(loss)

  return loss
    

def balanced_cross_entropy(beta):
    print('balanced_cross_entropy')
    def convert_to_logits(y_pred):
        # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

        return tf.log(y_pred / (1 - y_pred))

    def loss(y_true, y_pred):
        '''
        beta = tf.reduce_sum(1 - y_true) / (64 * 128 * 128)
        print('------------=================------------------')
        tf.print(beta, [beta], "image values=")
        print(tf.print(beta, [beta], "image values="))
        print('------------2222222222222222------------------')
        sess = tf.Session()
        with sess.as_default():
            print_op = tf.print(beta)
            print(print_op)
        '''
        y_pred = convert_to_logits(y_pred)
        pos_weight = beta / (1 - beta)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)

        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss * (1 - beta))

    return loss


def VGG_16(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path='./vgg16_weights.h5'):
    inputs = Input(shape=(3,img_rows, img_cols))
    conv1 = Conv2D(64, (3, 3), activation='relu', name='conv1_1', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', name='conv1_2', padding='same')(conv1)
    pool1 = MaxPooling2D((2,2), strides=(2,2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', name='conv2_1', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', name='conv2_2', padding='same')(conv2)
    pool2 = MaxPooling2D((2,2), strides=(2,2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', name='conv3_1', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', name='conv3_2', padding='same')(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', name='conv3_3', padding='same')(conv3)
    pool3 = MaxPooling2D((2,2), strides=(2,2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', name='conv4_1', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', name='conv4_2', padding='same')(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', name='conv4_3', padding='same')(conv4)
    pool4 = MaxPooling2D((2,2), strides=(2,2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', name='conv5_1', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', name='conv5_2', padding='same')(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', name='conv5_3', padding='same')(conv5)
    
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    
    if bottom_crop == 0:
        conv10 = Cropping2D(cropping=((crop_margin, crop_margin),(crop_margin, crop_margin)))(conv10) # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
    else:
        conv10 = Cropping2D(cropping=((0, bottom_crop),(0, right_crop)))(conv10)  # remove reflected portion from the image for prediction
   
    model = Model(inputs=inputs, outputs=conv10)
    model.load_weights(weights_path, by_name=True)
    #loss=weighted_cross_entropy(3)
    #loss=['binary_crossentropy']
    model.compile(optimizer=Adam(lr=1e-5), loss=weighted_cross_entropy(3), metrics=[dice_coef])
    
    return model

def preprocess_output(imgs, img_rows, img_cols):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    imgs_p = imgs_p.astype('float32')
    imgs_p /= 255.  # scale masks to [0, 1]
    return imgs_p
"""
def preprocess_input(imgs, img_rows, img_cols, mean, std):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1]*3, img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
        imgs_p[i, 1] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
        imgs_p[i, 2] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
       
    imgs_p = imgs_p.astype('float32')
    imgs_p -= mean
    imgs_p /= std
    return imgs_p
    """
    
def preprocess_input(imgs, img_rows, img_cols, mean, std):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    print('preprocess_input')
    print(imgs_p.shape)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
        imgs_p[i, 1] = cv2.resize(imgs[i, 1], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
        imgs_p[i, 2] = cv2.resize(imgs[i, 2], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
       
    imgs_p = imgs_p.astype('float32')
    imgs_p -= mean
    imgs_p /= std
    return imgs_p

def generate_data(imgs, msks, iteration):
    # define data preparation
    datagen = ImageDataGenerator(
        rotation_range=50.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect')
      
    imgs = imgs[:,np.newaxis,:,:].astype('uint8')
    msks = msks[:,np.newaxis,:,:].astype('uint8')
        
    train = np.zeros((iteration*128, 1, imgs.shape[2], imgs.shape[3])).astype('uint8')
    mask = np.zeros((iteration*128, 1, msks.shape[2], msks.shape[3])).astype('uint8')
    
    print('Data Generating...')
    for samples in range(iteration): 
        for imags_batch in datagen.flow(imgs, batch_size=128, seed = samples): 
            break 
        for imgs_mask_batch in datagen.flow(msks, batch_size=128, seed = samples): 
            break
        train[samples*128:(samples+1)*128] = imags_batch
        mask[samples*128:(samples+1)*128] = imgs_mask_batch
    
    train = np.vstack([imgs, train])
    mask = np.vstack([msks, mask])
    mask = mask[:,:,30:98,30:98]
    
    train, mask = shuffle(train, mask, random_state=10)
    return train, mask
