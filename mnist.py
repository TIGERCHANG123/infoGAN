from __future__ import print_function
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

ubuntu_root='/home/tigerc/temp'
windows_root='D:/Automatic/SRTP/GAN'

class mnist_dataset():
    def __init__(self, file_path = windows_root+'/datasets', noise_dim = 100):
        mnist, meta = tfds.load('mnist', data_dir=file_path, download=False, as_supervised=True, with_info=True)
        print(meta)
        self.train_dataset=mnist['train']
        self.noise_dim = noise_dim
        return
    def parse(self, x, y):
        x=tf.cast(x, tf.float32)
        x=x/255*2-1.0
        return x
    def get_train_dataset(self):
        train_dataset = self.train_dataset.map(self.parse).shuffle(60000).batch(128)
        return train_dataset

class noise_generator():
    def __init__(self, noise_dim, auxi_dim, dict_len, batch_size):
        self.noise_dim = noise_dim
        self.auxi_dim = auxi_dim
        self.dict_len = dict_len
        self.batch_size = batch_size
    def get_noise(self):
        noise = tf.random.normal([self.batch_size, self.noise_dim])
        noise = tf.cast(noise, tf.float32)
        auxi_dict = np.random.multinomial(1, self.dict_len * [float(1.0 / self.dict_len)],
                                             size=[self.batch_size])
        auxi_dict = tf.one_hot(tf.convert_to_tensor(auxi_dict))
        auxi_continuous = tf.convert_to_tensor(np.random.uniform(-1, 1, size=(self.batch_size, self.auxi_dim - self.dict_len)))
        auxi_code = tf.concat([auxi_dict, auxi_continuous], axis = -1)
        concat = tf.concat([noise, auxi_code], axis=-1)
        return noise, auxi_code
    def get_fixed_noise(self):
        noise = tf.random.normal([self.batch_size, self.noise_dim])
        noise = tf.cast(noise, tf.float32)
        auxi_dict = np.random.multinomial(1, self.dict_len * [float(1.0 / self.dict_len)],
                                             size=[self.batch_size])
        auxi_dict = tf.one_hot(tf.convert_to_tensor(auxi_dict))
        auxi_continuous = tf.convert_to_tensor(np.random.uniform(-1, 1, size=(self.batch_size, self.auxi_dim - self.dict_len)))
        auxi_code = tf.concat([auxi_dict, auxi_continuous], axis = -1)
        concat = tf.concat([noise, auxi_code], axis=-1)
        return concat
