#-*- codeing=utf-8 -*-
#@time: 2020/11/8 20:49
#@Author: Shang-gang Lee

import tensorflow as tf
from dataset import dataset
from ResNet18model import ResNet18
from tensorflow.keras import optimizers
def train(trainloader,valloader,testloader,optimizers,model,epoch=10):

    total_loss,total_acc = [],[]
    for epoch_ in range(epoch):
        for step,(x,y) in enumerate(trainloader):
            with tf.GradientTape() as tape:
                # [b, 32, 32, 3] => [b, 10],前向传播
                logits = model(x)
                y_one_hot=tf.one_hot(y,depth=100)# [b]=>[b,100]
                #print('logits:',logits.shape,'y_onehot:',y_one_hot.shape)
                loss=tf.losses.categorical_crossentropy(y_one_hot,logits,from_logits=True)  #loss
                loss=tf.reduce_mean(loss)   #mean loss

            grads=tape.gradient(loss,model.trainable_variables)
            optimizers.apply_gradients(zip(grads,model.trainable_variables))

            if step%100==0:
                total_loss.append(loss)
                print('epoch {}/{}'.format(epoch_,epoch),'------step:',step,'-----loss:',float(loss))

        acc,total=0,0
        for x, y in testloader:
            acc,total=0,0
            logits = model(x)
            prob = tf.nn.softmax(logits, axis=1)
            y_pred = tf.argmax(prob, axis=1)
            y_pred = tf.cast(y_pred, dtype=tf.int32)
            currect = tf.equal(y_pred, y)
            acc += tf.reduce_sum(tf.cast(currect, dtype=tf.int32)).numpy()
            total += x.shape[0]

        print('evaluate acc', acc / total)
        total_acc.append(acc)

if __name__ == '__main__':
    trainloader, valloader, testloader=dataset(batch=32)
    model = ResNet18()
    optimizer = optimizers.Adam(lr=1e-3)
    train(trainloader,valloader,testloader,optimizer,model,epoch=20)