#-*- codeing=utf-8 -*-
#@time: 2020/11/8 19:32
#@Author: Shang-gang Lee

import tensorflow as tf
from tensorflow.keras import layers,Sequential
from tensorflow import keras

class BasicBlack(layers.Layer):
    def __init__(self,filter_num,stride=1):
        super(BasicBlack, self).__init__()
        self.conv1 = layers.Conv2D(filter_num,kernel_size=(3,3),strides=stride,padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num,kernel_size=(3,3),strides=1,padding='same')
        self.bn2 = layers.BatchNormalization()
        if stride!= 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num,kernel_size=(1,1),strides=stride))
            self.downsample.add(layers.BatchNormalization())
        else:
            self.downsample = lambda x : x

    def call(self, inputs,training=None):
        residual = self.downsample(inputs)

        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1)
        relu1 = self.relu(bn1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)

        add = layers.add([residual,bn2])
        out = self.relu(add)
        return out

# resnet=BasicBlack(64)
# x=tf.random.normal([1,32,32,64])
# out = resnet(x)
# print(out.shape)

class ResNet(keras.Model):
    def __init__(self,layer_num,num_classes): # layer_num=[2,2,2,2]
        super(ResNet, self).__init__()
        self.stem = Sequential([layers.Conv2D(64,kernel_size=(3,3),strides=1,padding='same'),
                               layers.BatchNormalization(),
                               layers.Activation('relu'),
                               layers.MaxPool2D(pool_size=(2,2),strides=1,padding='same')])

        self.layer1 = self.build_resblock(64,layer_num[0])
        self.layer2 = self.build_resblock(128,layer_num[1],stride=2)
        self.layer3 = self.build_resblock(256,layer_num[2],stride=2)
        self.layer4 = self.build_resblock(512, layer_num[3],stride=2)
        # output:[b,512,h,w]

        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)

    def __call__(self,inputs,training=None):
        x = self.stem(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        logits = self.fc(x) # [b,100]
        return logits

    def build_resblock(self,filter_num,blocks,stride=1):
        res_blocks = Sequential()
        res_blocks.add(BasicBlack(filter_num,stride))

        for _ in range(1,blocks):
            res_blocks.add(BasicBlack(filter_num,stride=1))

        return res_blocks

def ResNet18():
    return ResNet(layer_num=[2,2,2,2],num_classes=100)

# model = ResNet18()
# x = tf.random.normal([1,224,224,3])
# out = model(x)
# print(out.shape)
#
# model = ResNet18()
# model.build(input_shape=(None, 32, 32, 3))
# model.summary()  # 统计网络参数