#-*- codeing=utf-8 -*-
#@time: 2020/11/8 20:45
#@Author: Shang-gang Lee
import processing
import tensorflow as tf
from tensorflow.keras import datasets
from processing import processing

def dataset(batch):
    (x,y),(test_x,test_y)=datasets.cifar100.load_data()
    y=tf.squeeze(y,axis=1)
    test_y=tf.squeeze(test_y,axis=1)
    train_x,val_x=tf.split(x,num_or_size_splits=[45000,5000])
    train_y,val_y=tf.split(y,num_or_size_splits=[45000,5000])
    print('train_x:', train_x.shape, 'train_y:', train_y.shape,
          'val_x:', val_x.shape, 'val_y:', val_y.shape,
          'test_x:', test_x.shape, 'test_y:', test_y.shape)

    trainloader=tf.data.Dataset.from_tensor_slices((train_x,train_y))
    trainloader=trainloader.map(processing).shuffle(10000).batch(batch)

    valloader=tf.data.Dataset.from_tensor_slices((val_x,val_y))
    valloader=valloader.map(processing).batch(batch)

    testloader=tf.data.Dataset.from_tensor_slices((test_x,test_y))
    testloader=testloader.map(processing).batch(batch)

    return trainloader,valloader,testloader