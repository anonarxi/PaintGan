from keras.layers import Concatenate,Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image
import os
import keras as keras


def generaInit(imp,nbGen):

	i=0
	p2=64
	up=[]
	input = Input(shape=imp)
	while(i < nbGen):
	
		if(i!=0):
			up.append(Conv2D(p2, kernel_size=3, strides=2, padding='same')(up[i-1]))
			up[i] = LeakyReLU(alpha=0.2)(up[i])
			up[i] = BatchNormalization(momentum=0.8)(up[i])
		else:
			up.append(Conv2D(p2, kernel_size=3, strides=2, padding='same')(input))
			up[i] = LeakyReLU(alpha=0.2)(up[i])

		i=i+1
		p2=p2*2

	u=up[nbGen-1]
	i=i-2
	p2=p2/2
	while(i>-1):
		u = UpSampling2D(size=2)(u)
		u = Conv2D(int(p2), kernel_size=3, strides=1, padding='same', activation='relu')(u)
		u = BatchNormalization(momentum=0.8)(u)
		u = Concatenate()([u, up[i]])			
		p2=p2/2
		i=i-1
		

	u = UpSampling2D()(u)
	model = Conv2D(3, kernel_size=3, strides=1, padding='same', activation='tanh')(u)
		
	return Model(input, model)


def discriInit(image_shape,nbDiscri):    
	model = Sequential()
	x=2;
	i=0;
	y=32;
	while(i < nbDiscri):
		if(i>(nbDiscri-1)/2):
			x=1
		model.add(Conv2D(y, kernel_size=3, strides=x,input_shape=image_shape, padding='same'))
		if(i!=0):
			model.add(BatchNormalization(momentum=0.9))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.2))
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
	datsat, datmap = data['arr_0'], data['arr_1']
	discriminator.compile(loss='binary_crossentropy',
	optimizer=Adam(1.5e-4, 0.5), metrics=['accuracy'])
	random_input = Input(shape=datsat[0].shape)
	generated_image = generator(random_input)
	discriminator.trainable = False
	discri = discriminator(generated_image)
	combined = Model(random_input, discri)
	combined.compile(loss='binary_crossentropy',
	optimizer=Adam(1.5e-4, 0.5), metrics=['accuracy'])

	y_real = np.ones((batch, 1))
	y_fake = np.zeros((batch, 1))


	cnt = 1
	for epoch in range(epochs):
		idx = np.random.randint(0, datsat.shape[0], batch)
		x_real = datsat[idx]
 
		noise= np.random.normal(0, 1, (batch, 100))
		x_fake = generator.predict(datmap[idx])
 
		discriminator_metric_real = discriminator.train_on_batch(x_real, y_real)
		discriminator_metric_generated = discriminator.train_on_batch(x_fake, y_fake)
 
		discriminator_metric = 0.5 * np.add(discriminator_metric_real, discriminator_metric_generated)
		generator_metric = combined.train_on_batch(datmap[idx], y_real)
		if epoch % save == 0:
			save_images(cnt, datmap[idx],size,4,7)
			cnt += 1 
			print(str(epoch)+' epoch, Discriminator accuracy: '+str(100*  discriminator_metric[1])+', Generator accuracy: '+str(100 * generator_metric[1]))
		if epoch % 50 == 0:
			discriminator.save('discriminator.h5')
			generator.save('generator.h5')

	
	save_images(cnt, datmap[idx],size,4,7)
	discriminator.save('discriminator.h5')
	generator.save('generator.h5')
	
	
	
imageSize=256
image_shape = (imageSize, imageSize, 3)
if os.path.exists('discriminator.h5'):
	discriminator = keras.models.load_model('discriminator.h5')
else:
	discriminator = discriInit(image_shape,5)
		
if os.path.exists('generator.h5'):
	generator = keras.models.load_model('generator.h5')
else:
	generator = generaInit(image_shape,5)
    
	
train(32,7,4,10000,20,imageSize,np.load('maps.npz'))
