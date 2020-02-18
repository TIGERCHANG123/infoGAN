import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from time import sleep
import os
import tensorflow as tf

class draw:
  batch_list = []
  loss_list = []
  acc_list = []
  i = 0
  def __init__(self, pic_size, root, model_dataset):
    rcParams['figure.figsize']=pic_size, pic_size
    self.pic_path = root + '/temp_pic/' + model_dataset
    self.pic_save_path = root + '/temp_pic_save/' + model_dataset
    self.generated_pic_path = root + '/generated_pic/' + model_dataset

    if not (os.path.exists(self.pic_path)):
        os.makedirs(self.pic_path)
    if not (os.path.exists(self.pic_save_path)):
        os.makedirs(self.pic_save_path)
    if not (os.path.exists(self.generated_pic_path)):
        os.makedirs(self.generated_pic_path)

    self.fig = plt.figure(figsize=(12, 4))
    self.batch_list = []
    self.train_loss_list = []
    self.train_acc_list = []
    self.i = 0
  def add(self, train_log):
    if len(self.batch_list) != 0:
      self.i = self.batch_list[-1] + 1
    else:
      self.i = self.i+1
    self.batch_list.append(self.i)
    self.train_loss_list.append(train_log[0])
    self.train_acc_list.append(train_log[1])
  def add_history(self, history):
      if len(self.batch_list) != 0:
          self.i = self.batch_list[-1] + len(history)
      else:
          self.i = self.i + len(history)
      self.batch_list.append(self.i)
      self.train_loss_list.append(history['loss'])
      self.train_acc_list.append(history['accuracy'])
  def save(self):
    NetPath = self.pic_save_path
    np.save(NetPath+'/b.npy', np.array(self.batch_list))
    np.save(NetPath+'/train_loss.npy', np.array(self.train_loss_list))
    np.save(NetPath+'/train_acc.npy', np.array(self.train_acc_list))
  def load(self, NetPath):
    self.batch_list = np.load(NetPath+'/b.npy').tolist()
    self.train_loss_list = np.load(NetPath+'/train_loss.npy').tolist()
    self.train_acc_list = np.load(NetPath+'/train_acc.npy').tolist()
  def close(self, time):
      sleep(time)
      plt.close()
  def show(self):
    file_path = self.pic_path
    plt.clf()
    ax1 = self.fig.add_subplot(121)
    ax2 = self.fig.add_subplot(122)
    ax1.plot(self.batch_list, self.train_loss_list, label = 'train loss', color = 'red')
    ax2.plot(self.batch_list, self.train_acc_list, label = 'train acc', color = 'red')
    bbox_props = dict(boxstyle='round',fc='w', ec='k',lw=1)
    ax1.annotate("%s" % self.train_loss_list[-1], xy=(self.i, self.train_loss_list[-1]), xytext=(-20, -20), textcoords='offset points', bbox=bbox_props)
    plt.annotate("%s" % self.train_acc_list[-1], xy=(self.i, self.train_acc_list[-1]), xytext=(-20, -20), textcoords='offset points', bbox=bbox_props)
    ax1.set(xlabel='batches',ylabel='loss', title = 'gen_loss')
    ax2.set(xlabel='batches',ylabel='loss', title = 'disc_loss')

    plt.savefig(file_path+str(self.i) + '.png')
    # thread1 = Thread(target=self.close, args=(1,))
    # thread1.start()
    # plt.show()
  def show_image(self, image):
    plt.imshow(image)
    plt.show()

  def show_created_pic(self, generator, num_list, auxi_list, noise_generator):
    auxi_con_1, auxi_con_2 = auxi_list
    for i, num in enumerate(num_list):
      _, _, x = noise_generator.get_fixed_noise(num, auxi_con_1, auxi_con_2)
      y = generator(x)
      y=tf.squeeze(y)
      plt.subplot(1, len(num_list), i + 1)
      plt.imshow(y.numpy(), 'gray')
      plt.axis('off')
      plt.tight_layout()
    plt.show()
    return

  def save_created_pic(self, generator, num_list, auxi_list, noise_generator, epoch):
    auxi_con_1, auxi_con_2 = auxi_list
    for i in num_list:
      _, _, x = noise_generator.get_fixed_noise(i, auxi_con_1, auxi_con_2)
      y = generator(x)
      y = tf.squeeze(y)
      plt.imsave(self.generated_pic_path + '/{}_{}.png'.format(epoch, i), y.numpy())

    return