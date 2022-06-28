from numpy import load
from numpy import zeros
from numpy import amax
import tensorflow as tf
from numpy import ones
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from numpy import asarray
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from matplotlib import pyplot

def define_encoder_block(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	return g
 
# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add upsampling layer
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	g = Concatenate()([g, skip_in])
	# relu activation
	g = Activation('relu')(g)
	return g

def define_generator(image_shape=(128,128,3)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# encoder model
	e1 = define_encoder_block(in_image, 32, batchnorm=False)
	e2 = define_encoder_block(e1, 64)
	e3 = define_encoder_block(e2, 128)
	e4 = define_encoder_block(e3, 256)
	e5 = define_encoder_block(e4, 256)
	e6 = define_encoder_block(e5, 256)
	e7 = define_encoder_block(e6, 256)
	# bottleneck, no batch norm and relu
	b = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)
	
	g = Dense(512,activation="relu")(b)
	out_image = Dense(6, activation="softmax")(g)

	# define model
	model = Model(in_image, out_image)
	return model
	
	


def generate_real_samples(X, y, n_samples=32):
	# unpack dataset
	#trainA, trainB = dataset
	# choose random instances
	ix = randint(0, 14020, n_samples)
	# retrieve selected images
	Xs, ys = X[ix], y[ix]
	# generate 'real' class labels (1)
	#y = ones((n_samples, patch_shape, patch_shape, 1))
	return Xs, ys




def classification_loss(y_actual,y_pred):
    cce = tf.keras.losses.CategoricalCrossentropy()
    act_loss = cce(y_actual, y_pred)
    print(act_loss)
    return act_loss 


#generate classifying model and compile it
gen = define_generator(image_shape=(128,128,3))
opt = Adam(learning_rate=0.0002, beta_1=0.5)
gen.compile(loss=['categorical_crossentropy'], optimizer=opt, metrics=['accuracy'])

# load data from npz format
X = load('EnvX128.npz')
Y = load('EnvY.npz')
X = X['arr_0']
X = (X - 127.5) / 127.5
print(X.max())
Y = Y['arr_0']
x, y = generate_real_samples(X, Y, 32)


#train model
for i in range(500*3):
    x, y = generate_real_samples(X, Y, 32)
    y = y.reshape(32,1,1,6)
    #pred = gen(x)
    #z = asarray(pred)
    #z = amax(z, axis=3)
    #z = z.reshape(1,32,1)
    #print(z.shape)
    print(y.shape)
    if i % 50 == 0:
        print(gen.evaluate(x, y))
    gen.train_on_batch(x, y)
    
gen.save('clasifying_model042622')

    
    
#gen = define_generator(image_shape=(128,128,3))
#print(gen.summary())
