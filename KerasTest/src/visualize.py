import matplotlib.pyplot as plt
import numpy
from scipy.misc import imsave
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras import backend
import theano
import tensorflow as tf
import time
import os
import h5py


img_width = 28
img_height = 28


#name of layer to visualize 
#TODO: create array of layers you would like to visualize and iterate over them
layer_name = 'dense_2'

#load model and weights
model = load_model('mnist_model1.h5')
print('loaded model')
model.load_weights('mnist_weights.h5')
print('loaded weights')
#print model summary
#model.summary()


#start with initial input, obviously
input_img = model.layers[0].input


for z in range(0,-20):
	print(z, " ------ ", model.layers[z])

layer_output = model.layers[-1].output


learning_rate = 500

#arbitrarily select filter 10 and class 3
class_index = 3 

print('Processing filter %d' % class_index)
start_time = time.time()
print("start time is: %d" % start_time)
#build a loss function that maximizes the activation of each filter of the layer considered
loss = layer_output[0, class_index]


#compute gradient of input picture wrt this loss
grads = backend.gradients(loss, input_img)[0]


#this function returns loss and grads given the input picture
iterate = backend.function([input_img, backend.learning_phase()],[loss,grads])

#set numpy's random seed for reproducibility
numpy.random.seed(333)

#start with gray image w/ random noise
#input_img_data = numpy.random.random((img_height, img_width, 1))
#distribute over range [-128,128]
#input_img_data = (input_img_data - 0.5) * 20 + 128
input_img_data = numpy.random.normal(0,10,(1,)+model.input_shape[1:])


print('by the way, the model input shape is: %d', model.input_shape[1:])

#run gradient ascent for 200 steps
for i in range(200):
	loss_value, grads_value = iterate([input_img_data,0]) #0 for test phase
	input_img_data += grads_value * learning_rate #apply gradient to image

	print('Current loss value: %d'% loss_value)



#print('Filter %d processed' % (filter_index))

#imsave('visualized_filter_10.png', img)











