import tensorflow as tf

class generator(tf.keras.Model):
  def __init__(self, image_size=[28, 28]):
    super(generator, self).__init__()
    self.dense_list = [16, 32, 64, len(image_size.flatten())]
    self.dense = [tf.keras.layers.Dense(i) for i in self.dense_list]
    self.dropout = [tf.keras.layers.dropout(0.5) for i in range(len(self.dense_list))]

  def call(self, x):
    for i in range(len(self.dense_list)):
      x = self.dense[i](x)
      x = self.dropout[i](x)
    x=tf.keras.layers.Reshape(x, [28, 28, 1])
    return x

class discriminator(tf.keras.Model):
  def __init__(self, input_shape=[28, 28]):
    super(discriminator, self).__init__()
    self.dense_list = [len(input_shape.flatten()), 64, 32, 16]
    self.dense = [tf.keras.layers.Dense(i) for i in self.dense_list]
    self.dropout = [tf.keras.layers.dropout(0.5) for i in range(len(self.dense_list))]

  def call(self, x, hidden):
    for i in range(len(self.dense_list)):
      x = self.dense[i](x)
      x = self.dropout[i](x)
    x = tf.keras.layers.Reshape(x, [28, 28, 1])
    return x



