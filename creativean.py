# -*- coding: utf-8 -*-
"""creativeAN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17SBi6W3y3ZXU8A5G9f40pqLLnmVMdHZr
"""

import tensorflow as tf
import glob
from numpy.random import randint
#import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import warnings
from IPython import display
from keras import initializers
from tensorflow.keras.optimizers import Adam
from keras.layers import Input, Dense, Dropout, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization

warnings.simplefilter('ignore')
from google.colab import drive
drive.mount('/content/drive')

data1 = np.load('/content/drive/My Drive/EnvX128.npz')
train_images = data1['arr_0']
#train_images = train_images[:5000]
print(train_images.shape)
print(train_images.max())


'''	
data2 = np.load('/content/drive/My Drive/128_faces_split2.npz')
train_images2 = data2['arr_0']
#train_images2 = train_images2[:6000]
print(train_images2.shape)

train_images = np.concatenate((train_images1, train_images2), axis=0)

data2 = np.load('/content/drive/My Drive/128_faces_split3.npz')
train_images2 = data2['arr_0']
#train_images2 = train_images2[:6000]
print(train_images2.shape)

train_images = np.concatenate((train_images, train_images2), axis=0)
'''

#numpy array in .npy of 128X128 images with 3 RGB channels
#train_images = np.load('/content/drive/My Drive/128_faces_split1.npz')
#train_images = train_images['arr_0']
print(train_images.shape)
plt.imshow(train_images[5].reshape(128, 128, 3))
print(train_images[5].reshape(128, 128, 3).max())

train_images = train_images.reshape(train_images.shape[0], 128, 128, 3).astype('float32')
#normalize images
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 16

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

img_shape = (128, 128, 3)
z_dim = 100
init = initializers.RandomNormal(mean=0.0, stddev=0.02)
opt2 = Adam(lr=0.0002, beta_1=0.5)


import tensorflow as tf

from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization,\
                                    Add, AveragePooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
!pip install git+https://www.github.com/keras-team/keras-contrib.git
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.initializers import RandomNormal
from tensorflow.keras import layers

BUFFER_SIZE = 400
EPOCHS = 100
LAMBDA = 100
DATASET = 'facades'
BATCH_SIZE = 8
IMG_WIDTH = 512
IMG_HEIGHT = 512
patch_size = 16
num_patches = (IMG_HEIGHT // patch_size) ** 2
projection_dim = 256
embed_dim = 64
num_heads = 2 
ff_dim = 32

assert IMG_WIDTH == IMG_HEIGHT, "image width and image height must have same dims"

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(x, filters, kernel_size, strides=2):
    x = layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.LeakyReLU()(x)


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn

def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out

def define_encoder_block(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv2D(n_filters, (4,4), strides=(1,1), padding='same')(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	return g

def make_generator_model():

    inputs = layers.Input(shape=(100))
    in2 = Dense(1024*256)(inputs)

    r2 = layers.Reshape((512,512,1))(in2)

    patches = Patches(patch_size)(r2)
    
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    
    x = TransformerBlock(256, num_heads, ff_dim)(encoded_patches)
    x = TransformerBlock(256, num_heads, ff_dim)(x)
    x = TransformerBlock(256, num_heads, ff_dim)(x)
    x = TransformerBlock(256, num_heads, ff_dim)(x)
    x = TransformerBlock(256, num_heads, ff_dim)(x)

    #p = layers.Reshape((8, 8, 256))(encoded_patches)

    rr = layers.Reshape((16, 16, 1024))(x)

    rr1 = layers.Reshape((16, 16, 1024))(in2)

    rc = Concatenate()([rr, rr1])

    x = layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same')(rc)
    x = layers.BatchNormalization(momentum=0.8, epsilon=0.00005)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    #x = residual_block(x, downsample=False, filters=512)

    x = layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8, epsilon=0.00005)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    #x = residual_block(x, downsample=False, filters=256)

    x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8, epsilon=0.00005)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    #x = residual_block(x, downsample=False, filters=64)

    x = layers.Conv2DTranspose(32, (5, 5), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8, epsilon=0.00005)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    #x = residual_block(x, downsample=False, filters=32)

    #x = layers.Conv2DTranspose(16, (5, 5), strides=(1, 1), padding='same')(x)
    #x = layers.BatchNormalization(momentum=0.8, epsilon=0.00005)(x)
    #x = layers.LeakyReLU(alpha=0.2)(x)

    #x = residual_block(x, downsample=False, filters=16)

    x = layers.Conv2D(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh')(x)

    return tf.keras.Model(inputs=inputs, outputs=x)



'''
def make_generator_model():
    model = tf.keras.Sequential()
    
    model.add(layers.Dense(256*8*8, input_shape=(100,), kernel_initializer=init))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((8, 8, 256)))
    
    model.add(layers.Conv2DTranspose(1024, (5, 5), strides=2, padding='same'))
    model.add(layers.BatchNormalization(momentum=0.8, epsilon=0.00005))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2DTranspose(512, (5, 5), strides=2, padding='same'))
    model.add(layers.BatchNormalization(momentum=0.8, epsilon=0.00005))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2DTranspose(256, (5, 5), strides=2, padding='same'))
    model.add(layers.BatchNormalization(momentum=0.8, epsilon=0.00005))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=2, padding='same'))
    model.add(layers.BatchNormalization(momentum=0.8, epsilon=0.00005))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=1, padding='same'))
    model.add(layers.BatchNormalization(momentum=0.8, epsilon=0.00005))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2DTranspose(32, (5, 5), strides=1, padding='same'))
    model.add(layers.BatchNormalization(momentum=0.8, epsilon=0.00005))
    model.add(layers.LeakyReLU(alpha=0.2))
     
    model.add(layers.Conv2D(3, (5, 5), strides=1, padding='same', activation='tanh'))


    return model
'''

generator = make_generator_model()

#generate noise
noise = tf.random.normal([1, 100])
#generate image
generated_image = generator(noise, training=False)
#check dimensions of generated image
print(generated_image.shape)
shape = generated_image.shape[1:4]
#show image
plt.imshow(np.array(generated_image).reshape(shape) * 127.5 + 127.5)

from keras.models import Sequential
#from keras.optimizers import Adam
#from tensorflow.keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.utils.vis_utils import plot_model

def make_discriminator_model(in_shape=(128,128,3)):
	model = Sequential()
	# normal
	model.add(Conv2D(16, (3,3), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(32, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(64, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
  
	model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
 
	model.add(Conv2D(512, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))

	# classifier
	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

'''
def make_discriminator_model():
    model = tf.keras.Sequential()

    model.add(Conv2D(64, (5, 5), strides=2, input_shape=img_shape, padding='same', kernel_initializer=init))
    model.add(BatchNormalization(momentum=0.8, epsilon=0.00005))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2D(128, (5, 5), strides=2, padding='same'))
    model.add(BatchNormalization(momentum=0.8, epsilon=0.00005))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2D(256, (5, 5), strides=2, padding='same'))
    model.add(BatchNormalization(momentum=0.8, epsilon=0.00005))
    model.add(LeakyReLU(alpha=0.2))
    #model.add(layers.Dropout(0.3))

    model.add(Conv2D(512, (5, 5), strides=1, padding='same'))
    model.add(BatchNormalization(momentum=0.8, epsilon=0.00005))
    model.add(LeakyReLU(alpha=0.2))
    #model.add(layers.Dropout(0.3))

    model.add(Conv2D(1024, (5, 5), strides=2, padding='same'))
    model.add(BatchNormalization(momentum=0.8, epsilon=0.00005))
    model.add(LeakyReLU(alpha=0.2))
    #model.add(layers.Dropout(0.3))
    
    model.add(Conv2D(2048, (5, 5), strides=2, padding='same'))
    model.add(BatchNormalization(momentum=0.8, epsilon=0.00005))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    
    return model
'''

#make discriminator from scratch
discriminator = make_discriminator_model()

# restore weights of generator and discriminator from checkpoint for further training 
'''
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = '/content/drive/My Drive/training_1'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

checkpoint.restore('/content/drive/My Drive/training_1/ckpt-54')
#checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
print(tf.train.latest_checkpoint(checkpoint_dir))
'''

#predict and print generated image using discriminator
decision = discriminator(generated_image)
print (decision)

gen = tf.keras.models.load_model('/content/drive/My Drive/clasifying_model042622')
gen.summary()

# This returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#calculates discriminator loss
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

#calculates generator loss
def generator_loss(fake_output, X):
    out = gen(X)
    c_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out,labels=(1.0/6)*tf.ones_like(out))) *.8
    return cross_entropy(tf.ones_like(fake_output), fake_output) + c_loss

# produces evalution updates for training
def evaluate(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    
    generated_images = generator(noise, training=False)

    real_output = discriminator(images, training=False)
    fake_output = discriminator(generated_images, training=False)
    print("Gen Loss")
    print(generator_loss(fake_output, images))
    print("disc Loss")
    print(discriminator_loss(real_output, fake_output))

def generate_real_samples(X, n_samples=32):
	# unpack dataset
	#trainA, trainB = dataset
	# choose random instances
	ix = randint(0, len(X), n_samples)
	# retrieve selected images
	Xs = X[ix]
	# generate 'real' class labels (1)
	#y = ones((n_samples, patch_shape, patch_shape, 1))
	return Xs



#initialiizes optimizers
generator_optimizer = tf.keras.optimizers.Adam(2e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4)

#creates checkpoint object
checkpoint_dir = '/content/drive/My Drive/training_1'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 125
noise_dim = 100
num_examples_to_generate = 16

# You will reuse this seed overtime (so it's easier)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

''' 
The loss is calculated for each of these models, and the gradients are 
used to update the generator and discriminator.
'''

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output, images)
      disc_loss = discriminator_loss(real_output, fake_output)
      

    
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))



# training procedure
def train(dataset, epochs):
  eo = 0
  i = 0
  for epoch in range(epochs):
    start = time.time()
    for image_batch in (dataset):
      i +=1
      #produces evaluation updates for each epoch
      if i %450 ==0:
        evaluate(image_batch)
      #imb = generate_real_samples(dataset, n_samples=16)
      train_step(image_batch)
      eo = epoch

    # Produce images for the GIF as you go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      #save model in .h5 file
      #generator.save('/content/drive/My Drive/1c_gen_simp_%03d.h5' % (epoch+1))
      #save checkpoint for training from this point
      #checkpoint.save(file_prefix = checkpoint_prefix)
      pass

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, :])
      #print(predictions[i, :, :, 0].shape)
      plt.axis('off')

  plt.savefig('/content/drive/My Drive/image_at_epoch_simp{:04d}.png'.format(epoch))
  plt.show()

train(train_dataset, EPOCHS)





