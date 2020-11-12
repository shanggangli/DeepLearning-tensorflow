#-*- codeing=utf-8 -*-
#@time: 2020/11/7 10:04
#@Author: Shang-gang Lee
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics
import datetime
#hyper parameter
batch=32
lr=1e-3
epoch=10

current_time=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir='logs' + current_time
summary_writer=tf.summary.create_file_writer(log_dir)

def dataset():
    (x,y),(x_test,y_test)=datasets.fashion_mnist.load_data()
    print(x.shape,y.shape)
    return (x,y),(x_test,y_test)

def preprocess(x,y):
    x=tf.cast(x,dtype=tf.float32)/255.  # normalize
    x=tf.reshape(x,[-1,28*28])  # x:[b,28,28]=>[b,28*28]
    y=tf.cast(y,dtype=tf.int32)
    return x,y

def dataloader(x,y,batch=batch):
    dataloader=tf.data.Dataset.from_tensor_slices((x,y))
    dataloader=dataloader.shuffle(1000).batch(batch).map(preprocess)
    return dataloader

def model():
    model=Sequential([
        layers.Dense(256,activation=tf.nn.relu), #[b,785] =>[b,256]
        layers.Dense(128, activation=tf.nn.relu), #[b,256] =>[b,128]
        layers.Dense(64, activation=tf.nn.relu), #[b,128] =>[b,64]
        layers.Dense(32, activation=tf.nn.relu),  # [b,64] =>[b,32]
        layers.Dense(10)  #[b,32] =>[b,10]
    ])
    model.build(input_shape=[None, 28 * 28])
    return model

def train(traindataloader,testdataloader,model):
    total_acur, loss_value = [], []

    # w = w - lr*grad
    optimizer=optimizers.Adam(lr=lr)
    for epoch_ in range(epoch):
        for step,(x,y) in enumerate(traindataloader):

            # x=tf.reshape(x,[-1,28*28])
            y = tf.one_hot(y,depth=10)
            with tf.GradientTape() as tape:

                logits=model(x)

                #loss=tf.reduce_mean(tf.losses.MSE(y,logits))    #mean
                loss2=tf.reduce_mean(tf.losses.categorical_crossentropy(y,logits,from_logits=True))
                #print(loss2)

            grad=tape.gradient(loss2,model.trainable_variables)
            optimizer.apply_gradients(zip(grad,model.trainable_variables)) #autogradient

            if step%100==0:
                loss_value.append(float(loss2))
                print('epoch:{}/{}'.format(epoch_,epoch),'step:',step,'     loss:',float(loss2))
                with summary_writer.as_default():
                    tf.summary.scalar('train_loss',float(loss2),step=100)

            if step % 100 == 0:
                total, acc = 0, 0
                for x, y in testdataloader:
                    logits = model(x)
                    prob = tf.nn.softmax(logits,axis=1)
                    y_pred = tf.argmax(prob,axis=1)
                    y_pred = tf.cast(y_pred,dtype=tf.int32)
                    currect = tf.equal(y_pred,y)
                    acc += tf.reduce_sum(tf.cast(currect,dtype=tf.int32)).numpy()
                    total += x.shape[0]

                print('evaluate acc',acc/total)
                total_acur.append(acc)

                with summary_writer.as_default():
                    tf.summary.scalar('test_acc',float(acc/total),step=100)

if __name__ == '__main__':
    (x, y), (x_test, y_test)=dataset()
    train_dataloader,test_dataloader=dataloader(x,y),dataloader(x_test, y_test)
    model=model()
    train(train_dataloader,test_dataloader,model)




