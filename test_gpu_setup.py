import tensorflow as tf
from tensorflow.python.client import device_lib


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


if __name__ == "__main__":
    print(get_available_gpus())
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    from keras import backend as K
    print(K.tensorflow_backend._get_available_gpus())

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    print(sess)

    print(device_lib.list_local_devices())
    print('----------------------')

