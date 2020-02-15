from __future__ import print_function
import tensorflow_datasets as tfds
import tensorflow as tf
ubuntu_root='/home/tigerc/temp'
windows_root='D:/Automatic/SRTP/GAN'

class mnist_dataset():
    def __init__(self, file_path = windows_root+'/datasets'):
        mnist, meta = tfds.load('mnist', data_dir=file_path, download=False, as_supervised=True, with_info=True)
        print(meta)
        self.train_dataset=mnist['train']
        return
    def parse(self, x, y):
        x=tf.cast(x, tf.float64)
        x=x/255-0.5
        return x
    def get_train_dataset(self):
        train_dataset = self.train_dataset.map(self.parse).shuffle(60000).batch(128)
        return train_dataset


