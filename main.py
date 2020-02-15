# -*- coding:utf-8 -*-
import os
import numpy as np
import random
import tensorflow as tf
from GAN import generator, discriminator
from show_pic import draw
from Train import train_one_epoch
from mnist import mnist_dataset
from evaluate import show_created_pic

ubuntu_root='/home/tigerc/temp'
windows_root='D:/Automatic/SRTP/GAN/temp'
model_dataset = 'translate_pt_to_en'
root = ubuntu_root

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
    if not (os.path.exists(root + '/temp_pic/' + model_dataset)):
        os.makedirs(root + '/temp_pic/' + model_dataset)
    if not (os.path.exists(root + '/temp_pic_save/' + model_dataset)):
        os.makedirs(root + '/temp_pic_save/' + model_dataset)
    if not(os.path.exists(root + '/temp_txt_save/'+model_dataset)):
        os.makedirs(root + '/temp_txt_save/'+model_dataset)
    pic = draw(10)
    if not(os.path.exists(root+'/temp_txt_save/'+model_dataset+'/validation.txt')):
        txt = open(root+'/temp_txt_save/'+model_dataset+'/validation.txt','w')
    else:
        txt = open(root+'/temp_txt_save/'+model_dataset+'/validation.txt', 'a')
    dataset = mnist_dataset()
    train_dataset = dataset.train_dataset()

    optimizer = tf.keras.optimizers.Adam()
    generator = generator()
    discriminator = discriminator()

    checkpoint_path = root + '/temp_model_save/' + model_dataset
    ckpt = tf.train.Checkpoint(optimizer=optimizer,generator=generator, discriminator=discriminator)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    # 如果检查点存在，则恢复最新的检查点。
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    train = train_one_epoch(generator=generator, discriminator=discriminator, train_dataset=train_dataset,
              optimizer=optimizer, metrics=[train_loss, train_accuracy])
    for epoch in range(10):
        train.train(epoch=epoch, pic=pic)
        pic.show(root+'/temp_pic/' + model_dataset + '/pic')
        if (epoch + 1) % 5 == 0:
            ckpt_manager.save()
    x = tf.convert_to_tensor(np.random.rand(128))
    show_created_pic(generator, x)
    return

if __name__ == '__main__':
    main()