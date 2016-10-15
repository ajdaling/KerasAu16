import matplotlib.pyplot as plt
import numpy
from scipy.misc import imsave
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib import cm
from keras.models import load_model
from keras.utils import np_utils
from keras import backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
import scipy
import theano
import tensorflow as tf
import time
import os
import h5py


img_width = 28
img_height = 28

def deprocess_image(x):

	x -= x.mean()
	x /= (x.std() + 1e-5)
	x *= 0.1
		
	#clip to [0,1]
	x += 0.5
	x = numpy.clip(x,0,1)

	#convert to grayscale
	x *= 255	
	x = x.transpose((1,2,0)) # change from (channel,height,width) to (height,width,channel) for conversion to png
	x = numpy.clip(x,0,255).astype('uint8')

	return x	

#name of layer to visualize 
#TODO: create array of layers you would like to visualize and iterate over them


#load model and weights
model = load_model('mnist_model1.h5')
#recreate model without top
'''
batch_size = 128
nb_classes = 10
nb_epoch = 12
kernel_size = (3,3)
nb_filters = 32
pool_size = (2,2)
input_shape = (1,28,28)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
model.compile(loss='categorical_crossentropy',
             optimizer='adadelta',
             metrics=['accuracy'])

'''
print('loaded model')
model.load_weights('mnist_weights.h5')
#have to load weights manually to exclude top layers
'''
weights_path = ('mnist_weights.h5')
f = h5py.File(weights_path)
for k in range(12):
	if k >- len(model.layers):
		break
	g = f['layer_{}'.format(k)]
	weights = [g['param_{}'.format(p)] for p in range(12)]
	model.layers[k].set_weights(weights)
f.close()
'''
print('loaded weights')
#print model summary
#model.summary()
step = 0.5

layer_dict = dict([(layer.name, layer) for layer in model.layers])
layer_name = 'convolution2d_2' # don't forget to change the loss function

#start with initial input, obviously
input_img = model.layers[0].input

#for z in range(-12,0):
#	print(z, " ------ ", model.layers[z])

layer_output = layer_dict[layer_name].output


for n in range(20,30):
	filter_index = n
	print('Processing filter %d' % filter_index)
	start_time = time.time()
	#build a loss function that maximizes the activation of each filter of the layer considered
	#loss = backend.mean(layer_output[:, filter_index])
	loss = backend.mean(layer_output[:,filter_index,:,:]) # for non-dense layers

	#compute gradient of input picture wrt this loss
	grads = backend.gradients(loss, input_img)[0]
	#normalize gradient to avoid extremes
	grads /= (backend.sqrt(backend.mean(backend.square(grads))) + 1e-5)

	#this function returns loss and grads given the input picture
	iterate = backend.function([input_img, backend.learning_phase()],[loss,grads])

	#set numpy's random seed for reproducibility
	numpy.random.seed(333)

	#start with gray image w/ random noise
	input_img_data = numpy.random.random((1,1,img_width, img_height))*20+128

	#run gradient ascent for 200 steps
	for i in range(1000):
		loss_value, grads_value = iterate([input_img_data, 0]) #0 for test phase
		input_img_data += grads_value * step #apply gradient to image
	
		print('Current loss value: %d'% loss_value)


	#if loss_value > 0:
	img = deprocess_image(input_img_data[0])
	#kept_filters.append((img,loss_value))


	#decode resulting input image 
	#print(input_img_data[0])
	#img = deprocess_image(input_img_data[0])
	end_time = time.time()
	print('Filter %d processed in %ds' % (filter_index, end_time-start_time))

	scipy.misc.toimage(img[:,:,0]).save('mnist_cnn_layer_%s_filter_%d.png' %(layer_name, filter_index))

#imsave('visualized_filter_3.png', img)

#sort filters by loss so that only best nxn filters are shown
#kept_filters.sort(key=lambda x: x[1], reverse = True)
#kept_filters = kept_filters[:n*n]

#n = 1
#margin = 5
#width = img_width*n + margin*(n-1)
#height = img_height*n + margin*(n-1)
#stitched_filters = numpy.zeros((width,height,1))

#for i in range(n):
#	for j in range(n):
#		img, loss = kept_filters[0]
#		stitched_filters[(img_width + margin) * i: (img_width + margin)*i +img_width, (img_height + margin)*j: (img_height+margin)*j+img_height,:] = img
#

#scipy.misc.toimage(stitched_filters[:,:,0]).save('test_image1.png')
#imsave('stitched_filters%dx%d.png' % (n, n), stitched_filters)












