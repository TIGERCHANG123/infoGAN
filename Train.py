import tensorflow as tf

ubuntu_root='/home/tigerc/temp'
windows_root='D:/Automatic/SRTP/GAN/temp'
model_dataset = 'translate_pt_to_en'
root = ubuntu_root

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
    def __init__(self, generator, discriminator, train_dataset, optimizers, metrics, batch_size, noise_dim):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer, self.discriminator_optimizer = optimizers
        self.train_loss, self.train_accuracy = metrics
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.noise_dim=noise_dim
    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, ), dtype=tf.float64),
    ])
    def train_step(self, images):
        noise = tf.random.normal([self.batch_size, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
    def train(self, epoch,  pic):
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.val_loss.reset_states()
        self.val_accuracy.reset_states()
        # inp -> portuguese, tar -> english
        for (batch, (inp, tar)) in enumerate(self.train_dataset):
            self.train_step(inp=inp, tar=tar)
            pic.add([self.train_loss.result().numpy(), self.train_accuracy.result().numpy()])
            pic.save(root + '/temp_pic_save/' + model_dataset)
        accuracy = 0.0
        loss = 0.0
        t=0
        for (val_batch, (inp, tar)) in enumerate(self.val_dataset):
            self.val_step(inp, tar)
            loss += self.val_loss.result()
            accuracy += self.val_accuracy.result()
            t+=1
        print('epoch: {}, loss: {}, accuracy: {}'.format(epoch, self.val_loss.result()/t, self.val_accuracy.result()/t))