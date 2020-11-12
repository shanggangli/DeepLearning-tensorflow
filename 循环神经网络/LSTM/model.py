#-*- codeing=utf-8 -*-
#@time: 2020/11/9 20:05
#@Author: Shang-gang Lee
import tensorflow as tf
from tensorflow.keras import optimizers,layers,metrics,Sequential
from tensorflow import  keras

max_len = 80
batch = 32
class MyLSTM(keras.Model):
    def __init__(self,units,total_words,embedding_len,input_length):
        super(MyLSTM, self).__init__()
        # [b,64]
        self.h0 = [tf.zeros([batch,units]),tf.zeros([batch,units])]
        self.h1 = [tf.zeros([batch,units]),tf.zeros([batch,units])]
        # transform text to embedding representation
        # [b,80]=>[b,80,100]
        self.embedding = layers.Embedding(total_words,embedding_len,input_length=input_length)
        # [b,80,100]

        self.lstm_cell0 = layers.LSTMCell(units)
        self.lstm_cell1 = layers.LSTMCell(units)

        # fc [b,80,100] =>[b,64] => [b,1]
        self.fc = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs: [b,80]
        :param training:
        :param mask:
        :return:    [b,1]
        """
        x = self.embedding(inputs)

        # rnn cell
        # x*Wxh + h*Whh
        h0 = self.h0
        h1 = self.h1
        for word in tf.unstack(x, axis=1):
            out0, h0 = self.lstm_cell0(word,h0,training)
            out1, h1 = self.lstm_cell1(out0,h1,training)
            h0 = h1 # update hidden

        # out [b,64]
        x = self.fc(out1)
        prob = tf.sigmoid(x)
        return prob