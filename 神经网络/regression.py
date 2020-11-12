#-*- codeing=utf-8 -*-
#@time: 2020/11/6 20:47
#@Author: Shang-gang Lee

import tensorflow as tf
from tensorflow.keras import datasets
import matplotlib.pyplot as plt

#hyper parameter
batch=32
lr=1e-3
epoch=10

def dataset():
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    print('x:', x.shape, 'y:', y.shape, 'x test:', x_test.shape, 'y test:', y_test)
    return (x, y), (x_test, y_test)

def preprocess(x,y):
    x=tf.cast(x,dtype=tf.float32)/255.  # normalize
    x=tf.reshape(x,[-1,28*28])  # x:[b,28,28]=>[b,28*28]
    y=tf.cast(y,dtype=tf.int32)
    y=tf.one_hot(y,depth=10)    # one hot
    return x,y

def dataloader(x,y,batch=batch):
    dataloader=tf.data.Dataset.from_tensor_slices((x,y))
    dataloader=dataloader.shuffle(1000).batch(batch).map(preprocess)
    return dataloader

def train(traindataloder,testdataloader,lr,epoch=10):
    total_acur,loss_value=[],[]

    # 784 => 512
    w1, b1 = tf.Variable(tf.random.normal([784, 256], stddev=0.1)), tf.Variable(tf.zeros([256]))
    # 512 => 256
    w2, b2 = tf.Variable(tf.random.normal([256, 128], stddev=0.1)), tf.Variable(tf.zeros([128]))
    # 256 => 10
    w3, b3 = tf.Variable(tf.random.normal([128, 10], stddev=0.1)), tf.Variable(tf.zeros([10]))
    for epoch_ in range(epoch):
        for step,(x,y) in enumerate(traindataloder):

            with tf.GradientTape() as tape:
                # layer1.
                h1 = x @ w1 + b1
                h1 = tf.nn.relu(h1)
                # layer2
                h2 = h1 @ w2 + b2
                h2 = tf.nn.relu(h2)
                # output
                out = h2 @ w3 + b3
                # out = tf.nn.relu(out)

                # compute loss
                loss = tf.square(y-out)
                loss =tf.reduce_mean(loss)

            grads = tape.gradient(loss,[w1,b1,w2,b2,w3,b3])
            for p,g in zip([w1,b1,w2,b2,w3,b3],grads):
                p.assign_sub(lr * g)

            if step%100==0:
                loss_value.append(float(loss))
                print('epoch:{}/{}'.format(epoch_,epoch),'step:',step,'     loss:',float(loss))

            if step % 100 == 0:
                total,acc=0,0
                for x, y in testdataloader:
                    # layer1.
                    h1 = x @ w1 + b1
                    h1 = tf.nn.relu(h1)
                    # layer2
                    h2 = h1 @ w2 + b2
                    h2 = tf.nn.relu(h2)
                    # output
                    out = h2 @ w3 + b3
                    # out = tf.nn.relu(out)

                    out = tf.argmax(out,axis=1)
                    y = tf.argmax(y,axis=1)
                    currect=tf.equal(out,y)
                    acc += tf.reduce_sum(tf.cast(currect,dtype=tf.int32)).numpy()
                    total+=x.shape[0]

                print('evaluate acc',acc/total)
                total_acur.append(acc)

    plt.figure()
    x=[i*100 for i in range(len(loss_value))]
    plt.plot(x,loss_value,color='C0',marker='s',label='train')
    plt.ylabel('MSE')
    plt.xlabel('Step')
    plt.legend()
    plt.savefig('train.svg')

    plt.figure()
    plt.plot(x,total_acur,color='C1',marker='s',label='test')
    plt.ylabel('准确率')
    plt.xlabel('Step')
    plt.legend()
    plt.savefig('test.svg')

if __name__ == '__main__':
    (x, y), (x_test, y_test)=dataset()
    train_dataloader,test_dataloader=dataloader(x,y),dataloader(x_test, y_test)
    train(train_dataloader,test_dataloader,lr=lr,epoch=10)
