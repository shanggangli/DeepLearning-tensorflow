#-*- codeing=utf-8 -*-
#@time: 2020/11/8 20:41
#@Author: Shang-gang Lee

import tensorflow as tf
def processing(x,y):
    x=tf.cast(x,dtype=tf.float32)/255.
    y=tf.cast(y,dtype=tf.int32)
    return x,y
