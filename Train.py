import tensorflow as tf
import numpy as np

class train_one_epoch():
    def __init__(self, model, train_dataset, optimizers, metrics, noise_generator):
        self.generator, self.discriminator, self.auxiliary = model
        self.generator_optimizer, self.discriminator_optimizer, self.auxiliary_optimizer = optimizers
        self.gen_loss, self.disauxi_loss, self.auxi_loss = metrics
        self.train_dataset = train_dataset
        self.noise_genrator = noise_generator

    def discriminator_loss(self, real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    def auxiliary_loss(self, auxi_pred, auxi_real):
        dict_len = self.noise_genrator.dict_len
        auxi_dict_pred = auxi_pred[:, :dict_len]
        auxi_dict_real = auxi_real[:, :dict_len]
        auxi_loss_dict = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=auxi_dict_pred, labels=auxi_dict_real))
        auxi_continuous_pred = auxi_pred[:, dict_len:]
        auxi_continuous_real = auxi_real[:, dict_len:]
        auxi_loss_continuouts = tf.reduce_mean(
            tf.reduce_sum(tf.square(auxi_continuous_pred - auxi_continuous_real), axis=1))

        return auxi_loss_dict + auxi_loss_continuouts

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, 62), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 12), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32),
    ])
    def train_step(self, noise, auxi_real, images):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disauxi_tape, tf.GradientTape() as auxi_tape:
            g_input = tf.concat([noise, auxi_real], axis=-1)
            generated_images = self.generator(g_input, training=True)
            _, real_output = self.discriminator(images, training=True)
            auxi_input, fake_output = self.discriminator(generated_images, training=True)
            auxi_output = self.auxiliary(auxi_input)

            gen_loss = self.generator_loss(fake_output)
            disauxi_loss = self.discriminator_loss(real_output, fake_output)
            auxi_loss = self.auxiliary_loss(auxi_output, auxi_real)

        self.gen_loss(gen_loss)
        self.disauxi_loss(disauxi_loss)
        self.auxi_loss(auxi_loss)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disauxi_tape.gradient(disauxi_loss, self.discriminator.trainable_variables)
        gradients_of_auxiliary = auxi_tape.gradient(auxi_loss, self.auxiliary.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        self.auxiliary_optimizer.apply_gradients(zip(gradients_of_auxiliary, self.auxiliary.trainable_variables))
    def train(self, epoch,  pic):
        self.gen_loss.reset_states()
        self.disauxi_loss.reset_states()

        for (batch, image) in enumerate(self.train_dataset):
            noise, auxi_code = self.noise_genrator.get_noise()
            self.train_step(noise, auxi_code, image)
            pic.add([self.gen_loss.result().numpy(), self.disauxi_loss.result().numpy(), self.auxi_loss.result()])
            # if batch % 500 == 0:
        print('epoch: {}, gen loss: {}, disc loss: {}, auxi loss: {}'.format(epoch, self.gen_loss.result(), self.disauxi_loss.result(), self.auxi_loss.result()))