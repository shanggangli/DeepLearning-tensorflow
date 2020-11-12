#-*- codeing=utf-8 -*-
#@time: 2020/11/8 13:27
#@Author: Shang-gang Lee
import tensorflow as tf
from tensorflow.keras import datasets,optimizers,metrics,layers,Sequential

def LeNet5():
    #LeNet5
    conv_layers=Sequential([
        layers.Conv2D(6,kernel_size=[5,5],strides=1,activation=tf.nn.relu,padding='valid'),
        layers.MaxPool2D(pool_size=[2,2]),
        layers.Conv2D(16,kernel_size=[5,5],strides=1,activation=tf.nn.relu,padding='valid'),
        layers.MaxPool2D(pool_size=[2,2])
    ])
    return conv_layers

def fullconnetion():
    fc=Sequential([
        layers.Dense(256,activation=tf.nn.relu),
        layers.Dropout(0.2),
        layers.Dense(128,activation=tf.nn.relu),
        layers.Dropout(0.2),
        layers.Dense(100)
    ])
    return fc