import tensorflow as tf
import numpy as np

sess = tf.Session()
with sess.as_default():
    print(type(tf.constant([1,2,3]).eval()))
    print(tf.constant([1,2,3]).eval())

data = np.load('./Step1_extract_dataset/augmentation_dataset/all-patients_train_mask.npz')
train = data['arr_0']
mask = data['arr_1']
print(train.shape)
print(mask.shape)
print(train[0].shape)
print(mask[0].shape)