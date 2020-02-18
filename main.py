# -*- coding:utf-8 -*-
import os
import tensorflow as tf
import numpy as np
from infoGAN import generator, discriminator, auxiliary
from show_pic import draw
from Train import train_one_epoch
from mnist import mnist_dataset, noise_generator

ubuntu_root='/home/tigerc/temp'
windows_root='D:/Automatic/SRTP/GAN/temp'
model_dataset = 'dcgan_mnist'
root = windows_root

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
    pic = draw(10, root, model_dataset)
    noise_dim = 62
    auxi_dim = 12
    dataset = mnist_dataset()
    train_dataset = dataset.get_train_dataset()

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    auxi_optimizer = tf.keras.optimizers.Adam(1e-4)

    generator_model = generator(input_shape=[noise_dim, ])
    discriminator_model = discriminator()
    auxiliary_model = auxiliary()

    checkpoint_path = root + '/temp_model_save/' + model_dataset
    ckpt = tf.train.Checkpoint(genetator_optimizers=generator_optimizer, discriminator_optimizer=discriminator_optimizer ,auxi_optimizer=auxi_optimizer,
                               generator=generator_model, discriminator=discriminator_model, auxiliary=auxiliary_model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    gen_loss = tf.keras.metrics.Mean(name='gen_loss')
    disc_loss = tf.keras.metrics.Mean(name='disc_loss')
    auxi_loss = tf.keras.metrics.Mean(name='auxi_loss')

    get_noise = noise_generator(noise_dim, auxi_dim, 10, 128)
    train = train_one_epoch(model=[generator_model, discriminator_model, auxiliary_model], train_dataset=train_dataset,
                            optimizers=[generator_optimizer, discriminator_optimizer, auxi_optimizer],
                            metrics=[gen_loss, disc_loss, auxi_loss], noise_generator=get_noise
                            )

    for epoch in range(50):
        train.train(epoch=epoch, pic=pic)
        pic.show()
        if (epoch + 1) % 5 == 0:
            ckpt_manager.save()
        pic.save_created_pic(generator_model, np.arange(10), [0.2, 0.1], noise_generator=get_noise, epoch=epoch)
    pic.show_created_pic(generator_model, np.arange(10), [0.2, 0.1], noise_generator=get_noise)
    return

if __name__ == '__main__':
    main()