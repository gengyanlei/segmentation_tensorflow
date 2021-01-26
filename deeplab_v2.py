"""
author is leilei
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib import slim


class Deeplab_v2:
    '''
    Deeplab_v2 ,
    some layers no bias and no relu ;
    slim.conv2d has relu and bias  need to be close
    slim.batch_norm has relu , need to be opened
    '''

    def __init__(self, class_number):
        self.class_number = class_number

    def upsampling(self, bottom, feature_map_size):
        # feature_map_size: int [h,w]
        return tf.image.resize_bilinear(bottom, size=feature_map_size)

    def ASPP(self, bottom, depth=256):
        # artous spatial pyramid pooling  V2 and V3 not same
        # rate=24
        atrous_pool_block1 = slim.conv2d(bottom, depth, [3, 3], rate=24, activation_fn=None)
        # rate=6
        atrous_pool_block6 = slim.conv2d(bottom, depth, [3, 3], rate=6, activation_fn=None)
        # rate=12
        atrous_pool_block12 = slim.conv2d(bottom, depth, [3, 3], rate=12, activation_fn=None)
        # reat=18
        atrous_pool_block18 = slim.conv2d(bottom, depth, [3, 3], rate=18, activation_fn=None)
        # N*H*W*C
        net = tf.add_n([atrous_pool_block1, atrous_pool_block6, atrous_pool_block12, atrous_pool_block18])

        return net

    def start(self, bottom, is_training):
        out = slim.conv2d(bottom, 64, 7, 2, padding='SAME', activation_fn=None, biases_initializer=None)
        out = slim.batch_norm(out, center=True, scale=True, activation_fn=slim.nn.relu, is_training=is_training)
        # out=slim.nn.relu(out)
        out = slim.max_pool2d(out, 3, 2, padding='SAME')
        return out

    def bottleneck(self, bottom, num_output, is_training, stride=1, rate=1, downsample=None):
        residual = bottom

        conv1 = slim.conv2d(bottom, num_output, 1, stride, activation_fn=None, biases_initializer=None)
        bn1 = slim.batch_norm(conv1, center=True, scale=True, activation_fn=slim.nn.relu, is_training=is_training)

        conv2 = slim.conv2d(bn1, num_output, 3, 1, rate=rate, activation_fn=None, biases_initializer=None)
        bn2 = slim.batch_norm(conv2, center=True, scale=True, activation_fn=slim.nn.relu, is_training=is_training)

        conv3 = slim.conv2d(bn2, num_output * 4, 1, 1, activation_fn=None, biases_initializer=None)
        bn3 = slim.batch_norm(conv3, center=True, scale=True, is_training=is_training)

        if downsample is not None:
            # need to the same scale stride=2
            residual = slim.conv2d(bottom, num_output * 4, 1, stride, activation_fn=None, biases_initializer=None)
            residual = slim.batch_norm(residual, center=True, scale=True, is_training=is_training)

        out = tf.add(bn3, residual)
        out = slim.nn.relu(out)
        return out

    def encoder(self, bottom, layers_num, is_training):
        '''
        layers_num: lsit , every block contain nums res_bottleneck ; eg: [3,4,23,3]
        '''
        out = self.start(bottom, is_training)

        assert len(layers_num) == 4, 'Error: not 4 blocks'

        out = self.bottleneck(out, 64, is_training, downsample=True)
        for i in range(1, layers_num[0]):
            out = self.bottleneck(out, 64, is_training)

        out = self.bottleneck(out, 128, is_training, stride=2, downsample=True)
        for j in range(1, layers_num[1]):
            out = self.bottleneck(out, 128, is_training)

        out = self.bottleneck(out, 256, is_training, rate=2, downsample=True)
        for k in range(1, layers_num[2]):
            out = self.bottleneck(out, 256, is_training, rate=2)

        out = self.bottleneck(out, 512, is_training, rate=4, downsample=True)
        for l in range(1, layers_num[3]):
            out = self.bottleneck(out, 512, is_training, rate=4)
        # aspp
        out = self.ASPP(out, depth=self.class_number)

        return out

    def build_G(self, image, is_training):
        with tf.name_scope('processing'):
            b, g, r = tf.split(image, 3, axis=3)
            image = tf.concat([
                b * 0.00390625,
                g * 0.00390625,
                r * 0.00390625], axis=3)
        self.score_s = self.encoder(image, layers_num=[3, 4, 23, 3], is_training=is_training)
        self.score = self.upsampling(self.score_s, tf.shape(image)[1:3])
        self.softmax = slim.nn.softmax(self.score + tf.constant(1e-4))
        self.pred = tf.argmax(self.softmax, axis=-1)