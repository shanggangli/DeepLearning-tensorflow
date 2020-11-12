#-*- codeing=utf-8 -*-
#@time: 2020/11/10 21:36
#@Author: Shang-gang Lee

import tensorflow as tf
from tensorflow.keras import optimizers,layers,Sequential
from tensorflow import keras
import numpy as np
from dataset import dataset

tf.random.set_seed(1)
np.random.seed(1)

# hyper parameters
lr=1e-3
h_dim=20
batch=32
epoch=10
# net
class autoencoder(keras.Model):
    def __init__(self,h_dim):
        super(autoencoder, self).__init__()

        self.encoder=Sequential([
            layers.Dense(256,activation=tf.nn.relu),
            layers.Dense(128,activation=tf.nn.relu),
            layers.Dense(h_dim)
        ])

        self.decoder=Sequential([
            layers.Dense(128,activation=tf.nn.relu),
            layers.Dense(256,activation=tf.nn.relu),
            layers.Dense(784)
        ])
    def call(self, inputs, training=None, mask=None):
            # [b,784] => [b,10]
            h=self.encoder(inputs)
            x_hat=self.decoder(h)
            return x_hat

model = autoencoder(h_dim=h_dim)
model.build(input_shape=(None,784))
model.summary()

trainloader,testloader = dataset()
optimizer=optimizers.Adam(lr=lr)

for epoch_ in range(epoch):
    for step,x in enumerate(trainloader):
        x=tf.reshape(x,[-1,784])
        with tf.GradientTape() as tape:
            x_rec_logits = model(x)

            rec_loss = tf.losses.binary_crossentropy(x,x_rec_logits,from_logits=True)
            rec_loss = tf.reduce_mean(rec_loss)

        grads = tape.gradient(rec_loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))

    if step % 100 == 0:
        print('epoch{}/{}'.format(epoch_,epoch),'----step:',step,'----loss',rec_loss)

