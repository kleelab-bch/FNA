"""
Author: Junbong Jang
Date: 5/3/2021

Resize stained images for FNA project

"""
import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
from tqdm import tqdm


root_path = '../../tensorflowAPI/research/object_detection/dataset_tools/assets/'

# train_path_suffix = 'masks_train/'
# valid_path_suffix = 'masks_valid/'
# test_path_suffix = 'masks_test/'

# train_path_suffix = 'stained_images_train_improved/'
# valid_path_suffix = 'stained_images_valid_improved/'
# test_path_suffix = 'stained_images_test_improved/'

train_root_path = root_path + train_path_suffix
valid_root_path = root_path + valid_path_suffix
test_root_path = root_path + test_path_suffix
img_format = '*.png'

def load_image(image_file):
    image_name = tf.strings.regex_replace(image_file, train_root_path, '')
    image_name = tf.strings.regex_replace(image_name, valid_root_path, '')
    image_name = tf.strings.regex_replace(image_name, test_root_path, '')

    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32)

    return (image, image_name)


def resize(input_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image


def save_image(save_path, img_prefix, inp, img_height, img_width):
    input_image = inp[0]
    image_name = img_prefix + inp[1].numpy()[0].decode("utf-8")
    input_image = resize(input_image[0], img_height, img_width)
    input_image = input_image.numpy()
    cv2.imwrite(save_path + image_name, cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR))


def save_images(dataset, img_prefix, save_path, img_height, img_width):
    data_num = dataset.cardinality().numpy()
    print(data_num)

    if os.path.isdir(save_path) == 0:
        os.makedirs(save_path)

    for index, inp in enumerate(tqdm(dataset.take(data_num))):
        save_image(save_path, img_prefix, inp, img_height, img_width)


def resize_and_save_images(img_height, img_width, img_prefix):
    masks_train_dataset_names = tf.data.Dataset.list_files(train_root_path + img_format)
    masks_valid_dataset_names = tf.data.Dataset.list_files(valid_root_path + img_format)
    masks_test_dataset_names = tf.data.Dataset.list_files(test_root_path + img_format)

    train_dataset = masks_train_dataset_names.map(load_image,
                                                  num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().batch(1)
    valid_dataset = masks_valid_dataset_names.map(load_image,
                                                  num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().batch(1)
    test_dataset = masks_test_dataset_names.map(load_image,
                                                num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().batch(1)

    save_images(train_dataset, img_prefix, root_path + 'resized_' + train_path_suffix, img_height, img_width)
    save_images(valid_dataset, img_prefix, root_path + 'resized_' + valid_path_suffix, img_height, img_width)
    save_images(test_dataset, img_prefix, root_path + 'resized_' + test_path_suffix, img_height, img_width)


if __name__ == "__main__":
    # resize_and_save_images(1792, 2560, '')
    resize_and_save_images(1944, 2592, 'stained_')