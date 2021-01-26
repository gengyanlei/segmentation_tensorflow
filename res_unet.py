"""
the author is leilei
"""
'''
链接到结构图片：
https://github.com/ogvalt/deep_residual_unet/blob/master/architecture.jpg?raw=true
'''
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.layers import xavier_initializer

class Res_Unet:
    def __init__(self, class_number):
        self.class_number = class_number

    def upsampling(self, bottom, feature_map_size):
        # feature_map_size: int [h,w]
        return tf.image.resize_bilinear(bottom, size=feature_map_size)

    def res_block(self, bottom, num_outputs, strides, is_training):
        '''
        also: slim.batch_norm(activation_fn=slim.nn.relu)
        '''
        out = slim.batch_norm(bottom, center=True, scale=True, is_training=is_training)
        out = slim.nn.relu(out)
        out = slim.conv2d(out, num_outputs[0], 3, strides[0], padding='SAME', activation_fn=None)
        out = slim.batch_norm(out, center=True, scale=True, is_training=is_training)
        out = slim.nn.relu(out)
        out = slim.conv2d(out, num_outputs[1], 3, strides[1], padding='SAME', activation_fn=None)

        shortcut = slim.conv2d(bottom, num_outputs[1], 1, strides[0], padding='SAME', activation_fn=None)
        shortcut = slim.batch_norm(shortcut, center=True, scale=True, is_training=is_training)

        output = tf.add(shortcut, out)
        return output

    def encoder(self, bottom, is_training):
        to_decoder = []
        out = slim.conv2d(bottom, 64, 3, 1, padding='SAME', activation_fn=None)
        out = slim.batch_norm(out, center=True, scale=True, is_training=is_training)
        out = slim.nn.relu(out)
        out = slim.conv2d(out, 64, 3, 1, padding='SAME', activation_fn=None)

        shortcut = slim.conv2d(bottom, 64, 1, 1, padding='SAME', activation_fn=None)
        shortcut = slim.batch_norm(shortcut, center=True, scale=True, is_training=is_training)

        output = tf.add(shortcut, out)
        to_decoder.append(output)

        output = self.res_block(output, num_outputs=[128, 128], strides=[2, 1], is_training)
        to_decoder.append(output)

        output = self.res_block(output, num_outputs=[256, 256], strides=[2, 1], is_training)
        to_decoder.append(output)

        return to_decoder

    def decoder(self, bottom, from_encoder, is_training):
        out = self.upsampling(bottom, tf.shape(from_encoder[2])[1:3])
        out = tf.concat([out, from_encoder[2]], axis=-1)
        out = self.res_block(out, num_outputs=[256, 256], strides=[1, 1], is_training)

        out = self.upsampling(out, tf.shape(from_encoder[1])[1:3])
        out = tf.concat([out, from_encoder[1]], axis=-1)
        out = self.res_block(out, num_outputs=[128, 128], strides=[1, 1], is_training)

        out = self.upsampling(out, tf.shape(from_encoder[0])[1:3])
        out = tf.concat([out, from_encoder[0]], axis=-1)
        output = self.res_block(out, num_outputs=[64, 64], strides=[1, 1], is_training)

        return output

    def build(self, image, is_training):
        with tf.name_scope('processing'):
            b, g, r = tf.split(image, 3, axis=3)
            image = tf.concat([
                b * 0.00390625,
                g * 0.00390625,
                r * 0.00390625], axis=3)
        self.to_decoder = self.encoder(image, is_training)

        self.middle = self.res_block(self.to_decoder[2], num_outputs=[512, 512], strides=[2, 1], is_training)

        self.output = self.decoder(self.middle, self.to_decoder, is_training)

        self.score = slim.conv2d(self.output, self.class_number, 1, 1, activation_fn=None)

        self.softmax = slim.nn.softmax(self.score + tf.constant(1e-4))

        self.pred = tf.argmax(self.softmax, axis=-1)
