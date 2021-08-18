import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras import backend as K


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


if __name__ == "__main__":
    print(tf.__version__)
    print(get_available_gpus())
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
