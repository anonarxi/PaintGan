from keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image
import os
import keras as keras


def generaInit(noise_size,nbGen):
	model = Sequential()
	
	model.add(Dense(4 * 4 * 256, activation='relu',input_dim=noise_size))
	model.add(Reshape((4, 4, 256)))
		
	for i in range(nbGen):
		model.add(UpSampling2D())
		model.add(Conv2D(256, kernel_size=3, padding='same'))
		model.add(BatchNormalization())
		model.add(Activation('relu'))

	model.summary()
	model.add(Conv2D(3, kernel_size=3, padding='same'))
	model.add(Activation('tanh'))
	input = Input(shape=(noise_size,))
    
	return Model(input, model(input))


def discriInit(image_shape,nbDiscri):    
	model = Sequential()
	x=2;
	i=0;
	y=32;
	while(i < nbDiscri):
		if(i>(nbDiscri-1)/2):
			x=1
		model.add(Conv2D(y, kernel_size=3, strides=x,input_shape=image_shape, padding='same'))
		model.add(BatchNormalization())
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.3))
		i=i+1
		y=y*2
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	input_image = Input(shape=image_shape)
	return Model(input_image, model(input_image))


def save_images(cnt, noise,img_size,ligne,colone):
	image_array = np.full((	4 + (ligne * (img_size + 4)),4 + (colone * (img_size + 4)), 3),255, dtype=np.uint8)
		
	generated_images = generator.predict(noise)
	generated_images = 0.5 * generated_images + 0.5
	
	image_count = 0
	for row in range(ligne):
		for col in range(colone):
			r = row * (img_size + 4) + 4
			c = col * (img_size + 4) + 4
			image_array[r:r + img_size, c:c +img_size] = generated_images[image_count] * 255
			image_count += 1
	output_path = 'output'
	if not os.path.exists(output_path):
		os.makedirs(output_path)
		
	filename = os.path.join(output_path, "trained-"+str(cnt)+".png")
	im = Image.fromarray(image_array)
	im.save(filename)


def train(batch,colonne,ligne,epochs,save,size,data):
	
	discriminator.compile(loss='binary_crossentropy',
	optimizer=Adam(1.5e-4, 0.5), metrics=['accuracy'])
	random_input = Input(shape=(100,))
	generated_image = generator(random_input)
	discriminator.trainable = False
	discri = discriminator(generated_image)
	combined = Model(random_input, discri)
	combined.compile(loss='binary_crossentropy',
	optimizer=Adam(1.5e-4, 0.5), metrics=['accuracy'])

	y_real = np.ones((batch, 1))
	y_fake = np.zeros((batch, 1))

	fixed_noise = np.random.normal(0, 1, (ligne * colonne, 100))

	cnt = 1
	for epoch in range(epochs):
		idx = np.random.randint(0, data.shape[0], batch)
		x_real = data[idx]
 
		noise= np.random.normal(0, 1, (batch, 100))
		x_fake = generator.predict(noise)
 
		discriminator_metric_real = discriminator.train_on_batch(x_real, y_real)
		discriminator_metric_generated = discriminator.train_on_batch(x_fake, y_fake)
 
		discriminator_metric = 0.5 * np.add(discriminator_metric_real, discriminator_metric_generated)
		generator_metric = combined.train_on_batch(noise, y_real)
		if epoch % save == 0:
			save_images(cnt, fixed_noise,size,4,7)
			cnt += 1 
			print(str(epoch)+' epoch, Discriminator accuracy: '+str(100*  discriminator_metric[1])+', Generator accuracy: '+str(100 * generator_metric[1]))
		if epoch % 50 == 0:
			discriminator.save('discriminator.h5')
			generator.save('generator.h5')

	
	save_images(cnt, fixed_noise,size,4,7)
	discriminator.save('discriminator.h5')
	generator.save('generator.h5')
	
	
	
imageSize=128
image_shape = (imageSize, imageSize, 3)
if os.path.exists('discriminator.h5'):
	discriminator = keras.models.load_model('discriminator.h5')
else:
	discriminator = discriInit(image_shape,5)
		
if os.path.exists('generator.h5'):
	generator = keras.models.load_model('generator.h5')
else:
	generator = generaInit(100,5)
    
	
train(32,7,4,10000,20,imageSize,np.load('chiness_paint.npy'))
