#-*- codeing=utf-8 -*-
#@time: 2020/11/9 16:23
#@Author: Shang-gang Lee

from dataset import dataset
from model import MyRNN
from tensorflow.keras import optimizers
import tensorflow as tf
units=64
total_words=10000
embedding_len=80
optimizer=optimizers.Adam(lr=1e-3)
epochs=10
input_len=80

def train():
    model = MyRNN(units,total_words,embedding_len,input_len)
    trainloader, testloader = dataset()
    model.compile(optimizer = optimizer,loss = tf.losses.BinaryCrossentropy(),metrics=['accuracy'])
    model.fit(trainloader,epochs=epochs,validation_data=testloader)

if __name__ == '__main__':
    train()