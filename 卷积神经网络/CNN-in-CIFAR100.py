#-*- codeing=utf-8 -*-
#@time: 2020/11/8 11:22
#@Author: Shang-gang Lee

import tensorflow as tf
from tensorflow.keras import datasets,optimizers,metrics,layers,Sequential

#hyper parameter
lr=1e-3
epoch=10
batch=128

def processing(x,y):
    x=tf.cast(x,dtype=tf.float32)/255.
    y=tf.cast(y,dtype=tf.int32)
    return x,y

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

def train(trainloader,valloader,testloader,optimizers,model,fc,epoch=10):
    #[1,2]+[3,4]=-1,2,3,4]
    variables=model.trainable_variables + fc.trainable_variables
    total_loss,total_acc = [],[]
    for epoch_ in range(epoch):
        for step,(x,y) in enumerate(trainloader):
            with tf.GradientTape() as tape:

                out=model(x) #[b,32,32,3] => [b,5,5,16]
                out=tf.reshape(out,[-1,5*5*16])# flatten
                logits=fc(out)  #full connetion

                y_one_hot=tf.one_hot(y,depth=100)# [b]=>[b,100]
                #print('logits:',logits.shape,'y_onehot:',y_one_hot.shape)
                loss=tf.losses.categorical_crossentropy(y_one_hot,logits,from_logits=True)  #loss
                loss=tf.reduce_mean(loss)   #mean loss

            grads=tape.gradient(loss,variables)
            optimizers.apply_gradients(zip(grads,variables))

            if step%100==0:
                total_loss.append(loss)
                print('epoch {}/{}'.format(epoch_,epoch),'------step:',step,'-----loss:',float(loss))

        acc,total=0,0
        for x, y in testloader:
            acc,total=0,0
            logits = model(x)
            logits = tf.reshape(logits,[-1,5*5*16])
            out = fc(logits)
            prob = tf.nn.softmax(out, axis=1)
            y_pred = tf.argmax(prob, axis=1)
            y_pred = tf.cast(y_pred, dtype=tf.int32)
            currect = tf.equal(y_pred, y)
            acc += tf.reduce_sum(tf.cast(currect, dtype=tf.int32)).numpy()
            total += x.shape[0]

        print('evaluate acc', acc / total)
        total_acc.append(acc)

if __name__ == '__main__':
    trainloader, valloader, testloader=dataset(batch=32)
    lenet5=LeNet5()
    lenet5.build(input_shape=[None,32,32,3])
    fc=fullconnetion()
    fc.build(input_shape=[None,5*5*16])
    optimizer=optimizers.Adam(lr=1e-3)
    train(trainloader,valloader,testloader,optimizer,lenet5,fc,epoch=10)





