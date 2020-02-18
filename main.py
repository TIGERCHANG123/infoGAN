# -*- coding:utf-8 -*-
import os
import tensorflow as tf
import numpy as np
from infoGAN import get_gan
from show_pic import draw
from Train import train_one_epoch
from mnist import mnist_dataset, noise_generator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

ubuntu_root='/home/tigerc'
windows_root='D:/Automatic/SRTP/GAN'
root = ubuntu_root
temp_root = root+'/temp'

def main(continue_train, train_time):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
    noise_dim = 62
    auxi_dim = 12
    generator_model, discriminator_model, auxiliary_model, model_name = get_gan([noise_dim, ], [28, 28, 1])
    dataset = mnist_dataset(root=root)
    noise_gen = noise_generator(noise_dim, auxi_dim, 10, 128)
    
    model_dataset = model_name + dataset.name
    train_dataset = dataset.get_train_dataset()
    
    pic = draw(10, root, model_dataset, train_time)
    
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    auxi_optimizer = tf.keras.optimizers.Adam(1e-4)

    gen_loss = tf.keras.metrics.Mean(name='gen_loss')
    disc_loss = tf.keras.metrics.Mean(name='disc_loss')
    auxi_loss = tf.keras.metrics.Mean(name='auxi_loss')
    
    checkpoint_path = temp_root + '/temp_model_save/' + model_dataset
    ckpt = tf.train.Checkpoint(genetator_optimizers=generator_optimizer, discriminator_optimizer=discriminator_optimizer ,auxi_optimizer=auxi_optimizer,
                               generator=generator_model, discriminator=discriminator_model, auxiliary=auxiliary_model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint and continue_train:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    train = train_one_epoch(model=[generator_model, discriminator_model, auxiliary_model], train_dataset=train_dataset,
                            optimizers=[generator_optimizer, discriminator_optimizer, auxi_optimizer],
                            metrics=[gen_loss, disc_loss, auxi_loss], noise_generator=noise_gen
                            )

    for epoch in range(50):
        train.train(epoch=epoch, pic=pic)
        pic.show()
        if (epoch + 1) % 5 == 0:
            ckpt_manager.save()
        pic.save_created_pic(generator_model, np.arange(10), [0.2, 0.1], noise_generator=noise_gen, epoch=epoch)
    pic.show_created_pic(generator_model, np.arange(10), [0.2, 0.1], noise_generator=noise_gen)
    return

if __name__ == '__main__':
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    main(continue_train=False, train_time=0)