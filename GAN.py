import tensorflow as tf
from tensorflow.keras import layers

class generator_block_Conv2DTranspose(tf.keras.Model):
  def __init__(self, channels, strides):
    super(generator_block_Conv2DTranspose, self).__init__()
    self.Conv2DTranspose = tf.keras.layers.Conv2DTranspose(channels, (5, 5), strides=strides, padding='same', use_bias=False)
    self.bn = tf.keras.layers.BatchNormalization()
    self.leakyRelu = tf.keras.layers.LeakyReLU()
  def call(self, x, output):
    if not output:
      x = self.Conv2DTranspose(x)
      x = self.bn(x)
      x = self.leakyRelu(x)
      return x
    else:
      x = self.Conv2DTranspose(x)
      return x
class generator_block_Dense(tf.keras.Model):
  def __init__(self, channels, input_shape):
    super(generator_block_Dense, self).__init__()
    self.dense=tf.keras.layers.Dense(channels, use_bias=False, input_shape=input_shape)
    self.bn = tf.keras.layers.BatchNormalization()
    self.leakyRelu = tf.keras.layers.LeakyReLU()
  def call(self, x):
    x = self.dense(x)
    x = self.bn(x)
    x = self.leakyRelu(x)
    return x

class generator(tf.keras.Model):
  def __init__(self, input_shape=(100, )):
    super(generator, self).__init__()
    self.front_layer = generator_block_Dense(7*7*256, input_shape)
    Conv2Dlist = [128, 64, 1]
    self.Conv2DTranspose=[]

    self.reshape = tf.keras.layers.Reshape((7, 7, 256))
    self.Conv2DTranspose.append(generator_block_Conv2DTranspose(Conv2Dlist[0], 1))
    self.Conv2DTranspose.append(generator_block_Conv2DTranspose(Conv2Dlist[1], 2))
    self.Conv2DTranspose.append(generator_block_Conv2DTranspose(Conv2Dlist[2], 2))

    self.flatten = tf.keras.layers.Flatten()
  def call(self, x):
    x = self.flatten(x)
    x = self.front_layer(x)
    x = self.reshape(x)
    for i in range(len(self.Conv2DTranspose) - 1):
      x = self.Conv2DTranspose[i](x, output=False)
    x = self.Conv2DTranspose[-1](x, output=True)
    return x

class discriminator(tf.keras.Model):
  def __init__(self, input_shape=[28, 28, 1]):
    super(discriminator, self).__init__()
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',input_shape=input_shape))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    self.model = model
  def call(self, x):
    return self.model(x)



