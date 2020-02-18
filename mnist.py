from __future__ import print_function
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

class mnist_dataset():
    def __init__(self, root, batch_size):
        file_path = root + '/datasets/tensorflow_datasets'
        mnist, meta = tfds.load('mnist', data_dir=file_path, download=False, as_supervised=True, with_info=True)
        print(meta)
        self.train_dataset=mnist['train']
        self.name = 'mnist'
        self.batch_size = batch_size
        return
    def parse(self, x, y):
        x=tf.cast(x, tf.float32)
        x=x/255*2-1.0
        y = tf.one_hot(y, depth=10)
        y = tf.cast(y, tf.float32)

        auxi_continuous = tf.convert_to_tensor(np.random.uniform(-1, 1, size=(2)))
        auxi_continuous = tf.cast(auxi_continuous, tf.float32)
        auxi_code = tf.concat([y, auxi_continuous], axis=0)

        return x, auxi_code
    def get_train_dataset(self):
        train_dataset = self.train_dataset.map(self.parse).shuffle(60000).batch(self.batch_size)
        return train_dataset

class noise_generator():
    def __init__(self, noise_dim, auxi_dim, dict_len, batch_size):
        self.noise_dim = noise_dim
        self.auxi_dim = auxi_dim
        self.dict_len = dict_len
        self.batch_size = batch_size
    def get_noise(self, batch_size):
        noise = tf.random.normal([batch_size, self.noise_dim])
        noise = tf.cast(noise, tf.float32)
        auxi_dict = np.random.multinomial(1, self.dict_len * [float(1.0 / self.dict_len)],size=[batch_size])
        auxi_dict = tf.convert_to_tensor(auxi_dict)
        auxi_dict = tf.cast(auxi_dict, tf.float32)
        auxi_continuous = tf.convert_to_tensor(np.random.uniform(-1, 1, size=(batch_size, self.auxi_dim - self.dict_len)))
        auxi_continuous = tf.cast(auxi_continuous, tf.float32)
        auxi_code = tf.concat([auxi_dict, auxi_continuous], axis = 1)
        concat = tf.concat([noise, auxi_code], axis=-1)
        return noise, auxi_code

    def get_fixed_noise(self, num, auxi_con_1, axi_con_2):
        noise = tf.random.normal([1, self.noise_dim])
        noise = tf.cast(noise, tf.float32)

        auxi_dict = np.array([num])
        auxi_dict = tf.convert_to_tensor(auxi_dict)
        auxi_dict = tf.one_hot(auxi_dict, depth=self.dict_len)
        auxi_dict = tf.cast(auxi_dict, tf.float32)
        auxi_continuous = np.array([auxi_con_1, axi_con_2])
        auxi_continuous = np.reshape(auxi_continuous, [1, -1])
        auxi_continuous = tf.cast(auxi_continuous, tf.float32)
        auxi_code = tf.concat([auxi_dict, auxi_continuous], axis=1)
        concat = tf.concat([noise, auxi_code], axis=-1)
        return noise, auxi_code, concat
