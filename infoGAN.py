import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class generator(tf.keras.Model):
  def __init__(self, gen_input_shape, img_shape):
    super(generator, self).__init__()
    self.gen_input_shape = gen_input_shape
    self.img_shape = img_shape

    self.model = tf.keras.Sequential()
    self.model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=gen_input_shape))
    self.model.add(layers.BatchNormalization())
    self.model.add(layers.ReLU())
    self.model.add(layers.Reshape((7, 7, 256)))

    self.model.add(layers.Conv2DTranspose(128, (5, 5), strides=2, padding='same', use_bias=False))
    self.model.add(layers.BatchNormalization())
    self.model.add(layers.ReLU())

    self.model.add(layers.Conv2DTranspose(64, (5, 5), strides=2, padding='same', use_bias=False))
    self.model.add(layers.BatchNormalization())
    self.model.add(layers.ReLU())

    self.model.add(layers.Conv2DTranspose(1, (5, 5), strides=1, padding='same', use_bias=False))
    self.model.add(layers.Activation(activation='tanh'))
  def call(self, x):
    return self.model(x)

class discriminator(tf.keras.Model):
  def __init__(self, input_shape):
    super(discriminator, self).__init__()
    self.conv_input = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',input_shape=input_shape)
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.leakyRelu1 = layers.LeakyReLU(0.2)
    self.dropout1 = layers.Dropout(0.3)

    self.conv2 = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')
    self.bn2 = tf.keras.layers.BatchNormalization()
    self.leakyRelu2 = layers.LeakyReLU(0.2)
    self.dropout2 = layers.Dropout(0.3)
    self.flatten = layers.Flatten()
    self.denseD = layers.Dense(1)
  def call(self, x):
    x = self.conv_input(x)
    x = self.bn1(x)
    x = self.leakyRelu1(x)
    x = self.dropout1(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.leakyRelu2(x)
    x = self.dropout2(x)
    x = self.flatten(x)

    y = self.denseD(x)
    return x, y

class auxiliary(tf.keras.Model):
  def __init__(self, hidden=128, auxi_dim=12):
    super(auxiliary, self).__init__()
    self.dense = layers.Dense(hidden)
    self.bn = tf.keras.layers.BatchNormalization()
    self.leakyRelu = tf.keras.layers.LeakyReLU(0.1)

    self.outputDense = tf.keras.layers.Dense(auxi_dim)
  def call(self, x):
    x = self.dense(x)
    x = self.bn(x)
    x = self.leakyRelu(x)
    x = self.outputDense(x)
    return x

def get_gan(gen_input_shape, img_shape):
  Generator = generator(gen_input_shape, img_shape)
  Discriminator = discriminator(img_shape)
  Auxiliary = auxiliary()
  gen_name = 'infoGAN'
  return Generator, Discriminator, Auxiliary, gen_name



