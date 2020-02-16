# -*- coding:utf-8 -*-
import os
import tensorflow as tf
from GAN import generator, discriminator
from show_pic import draw
from Train import train_one_epoch
from mnist import mnist_dataset

ubuntu_root='/home/tigerc/temp'
windows_root='D:/Automatic/SRTP/GAN/temp'
model_dataset = 'dcgan_mnist'
root = windows_root

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
    pic = draw(10, root, model_dataset)
    noise_dim = 128
    dataset = mnist_dataset()
    train_dataset = dataset.get_train_dataset()

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    generator_model = generator(input_shape=[noise_dim, ])
    discriminator_model = discriminator()

    checkpoint_path = root + '/temp_model_save/' + model_dataset
    ckpt = tf.train.Checkpoint(genetator_optimizers=generator_optimizer, discriminator_optimizer=discriminator_optimizer ,
                               generator=generator_model, discriminator=discriminator_model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    gen_loss = tf.keras.metrics.Mean(name='gen_loss')
    disc_loss = tf.keras.metrics.Mean(name='disc_loss')

    train = train_one_epoch(model=[generator_model, discriminator_model], train_dataset=train_dataset,
              optimizers=[generator_optimizer, discriminator_optimizer], metrics=[gen_loss, disc_loss])

    for epoch in range(50):
        train.train(epoch=epoch, pic=pic)
        pic.show()
        if (epoch + 1) % 5 == 0:
            ckpt_manager.save()
        pic.save_created_pic(generator_model, 8, noise_dim, epoch)
    pic.show_created_pic(generator_model, 8, noise_dim)
    return

if __name__ == '__main__':
    main()