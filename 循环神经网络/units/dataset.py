#-*- codeing=utf-8 -*-
#@time: 2020/11/9 16:22
#@Author: Shang-gang Lee

import tensorflow as tf
from tensorflow.keras import datasets,preprocessing

# hyper parameter
batch = 32
total_word = 10000
max_len = 80

def dataset(batch=32,total_word=10000,max_len=80):
    (x_train,y_train),(x_test,y_test) = datasets.imdb.load_data(num_words=total_word)   # load_data
    #[b,80]
    x_train = preprocessing.sequence.pad_sequences(x_train,maxlen=max_len)  # truncate and padding
    x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

    # x_train shape (25000, 80) y_train shape (25000,) x_test shape (25000, 80) y_test shape (25000,)
    # print("x_train shape",x_train.shape,'y_train shape',y_train.shape,'x_test shape',x_test.shape,'y_test shape',y_test.shape)

    # print('y_train max',tf.reduce_max(y_train))   # max:1
    # print('y_train min',tf.reduce_min(y_train))   # min:0

    trainloader = tf.data.Dataset.from_tensor_slices((x_train,y_train))
    trainloader = trainloader.shuffle(10000).batch(batch,drop_remainder=True)

    testloader = tf.data.Dataset.from_tensor_slices((x_test,y_test))
    testloader = testloader.batch(batch,drop_remainder=True)

    return trainloader,testloader


#data = dataset()