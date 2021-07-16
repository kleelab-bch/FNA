from __future__ import print_function
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import random
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, Conv2DTranspose,BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
#K.set_image_dim_ordering('th')  # Theano dimension ordering in this code
K.set_image_data_format('channels_first')
smooth = 1.

'''
8/30/2020

Used to stain the FNA images to compare performance with other staining models such as Pix2pix.
'''


def weighted_cross_entropy(beta):
  def convert_to_logits(y_pred):
      # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
      y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
      # logistic function but inverse
      # map probability from 0 to 1 ----> Real Number from negative infinity to positive infinity
      return tf.math.log(y_pred / (1 - y_pred))

  def loss(y_true, y_pred):
    y_pred = convert_to_logits(y_pred)
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=beta)

    # or reduce_sum and/or axis=-1
    return tf.reduce_mean(loss)

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

    conv10 = Conv2D(3, (1, 1), activation='sigmoid')(conv9)
    

    #if bottom_crop == 0:
    #   conv10 = Cropping2D(cropping=((crop_margin, crop_margin),(crop_margin, crop_margin)))(conv10) # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
    #else:
    if bottom_crop != 0:
        conv10 = Cropping2D(cropping=((0, bottom_crop),(0, right_crop)))(conv10)  # remove reflected portion from the image for prediction

    model = Model(inputs=inputs, outputs=conv10)
    model.load_weights(weights_path, by_name=True)
    
    model.compile(optimizer=Adam(lr=1e-5), loss=tf.keras.losses.MAE)
    
    return model

