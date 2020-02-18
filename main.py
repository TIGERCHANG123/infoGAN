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

def main(continue_train, train_time, model_version):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
    noise_dim = 62
    auxi_dim = 12
    batch_size = 128
    generator_model, discriminator_model, auxiliary_model, model_name = get_gan([noise_dim+auxi_dim, ], [28, 28, 1])
    dataset = mnist_dataset(root=root, batch_size=batch_size)
    noise_gen = noise_generator(noise_dim, auxi_dim, 10,batch_size)
    model_dataset = '{}_{}_{}'.format(model_name, dataset.name, model_version)

    print('model: {}\ndataset: {}\nbatch_size: {}\nnoise_dim: {}\nauxi_dim: {}\nmodel version: {}'.format(
        model_name, dataset.name, batch_size, noise_dim, auxi_dim, model_version))
    train_dataset = dataset.get_train_dataset()
    
    pic = draw(10, temp_root, model_dataset, train_time)
    
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
        pic.load()
        print('Latest checkpoint restored!!')

    train = train_one_epoch(model=[generator_model, discriminator_model, auxiliary_model], train_dataset=train_dataset,
                            optimizers=[generator_optimizer, discriminator_optimizer, auxi_optimizer],
                            metrics=[gen_loss, disc_loss, auxi_loss], noise_generator=noise_gen
                            )

    for epoch in range(100):
        train.train(epoch=epoch, pic=pic)
        pic.show()
        if (epoch + 1) % 5 == 0:
            ckpt_manager.save()
            pic.save()
        rdm_list = np.random.uniform(-1, 1, size=(1, 2))
        pic.save_created_pic(generator_model, np.arange(10), rdm_list[0], noise_generator=noise_gen, epoch=epoch)
    pic.show_created_pic(generator_model, np.arange(10), [0.2, 0.1], noise_generator=noise_gen)
    return

if __name__ == '__main__':
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    main(continue_train=True, train_time=1, model_version=1)