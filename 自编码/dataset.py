#-*- codeing=utf-8 -*-
#@time: 2020/11/10 21:49
#@Author: Shang-gang Lee
from tensorflow import keras
import tensorflow as tf
import numpy as np

batch=32
def dataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.
    # we do not need label
    trainloader = tf.data.Dataset.from_tensor_slices(x_train)
    trainloader = trainloader.shuffle(batch * 5).batch(batch)
    testloader = tf.data.Dataset.from_tensor_slices(x_test)
    testloader = testloader.batch(batch)

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    return trainloader,testloader

# dataset()
