import tensorflow as tf

def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    return cross_entropy(tf.ones_like(fake_output), fake_output)

class train_one_epoch():
    def __init__(self, model, train_dataset, optimizers, metrics, noise_dim):
        self.generator, self.discriminator = model
        self.generator_optimizer, self.discriminator_optimizer = optimizers
        self.gen_loss, self.disc_loss = metrics
        self.train_dataset = train_dataset
        self.noise_dim=noise_dim
    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, 128), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32),
    ])
    def train_step(self, noise, images):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)
        self.gen_loss(gen_loss)
        self.disc_loss(disc_loss)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
    def train(self, epoch,  pic):
        self.gen_loss.reset_states()
        self.disc_loss.reset_states()

        for (batch, (noise, image)) in enumerate(self.train_dataset):
            noise = tf.random.normal([image.shape[0], self.noise_dim])
            noise = tf.cast(noise, tf.float32)
            self.train_step(noise, image)
            pic.add([self.gen_loss.result().numpy(), self.disc_loss.result().numpy()])
            pic.save()
            # if batch % 500 == 0:
            print('epoch: {}, gen loss: {}, disc loss: {}'.format(epoch, self.gen_loss.result(), self.disc_loss.result()))