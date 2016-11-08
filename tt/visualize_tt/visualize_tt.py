import numpy
import scipy
from keras.models import load_model
from keras.utils import np_utils
from keras import backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
import scipy
import theano
import time
import h5py


'''
This program generates n 32x32 grayscale .png images. These images represent inputs
that would maximize a particular filter at a particular layer of the cnn (tt_cnn).
The network model and weights are loaded in as .h5 files generated after training the
cnn.

author: Alec Daling
date: 11/3/16

adapted from: 

	1. github.com/fchollet/keras/blob/
	2. ankivil.com/visualizing-deep...
	3. keras blog (fchollet's walkthrough)

Notes: 
	1. The loss function is dependent upon the type of layer you are 
		using. Both kinds of loss function are included, just comment
		one out.
	2. The name of the layer has to specified, as well as the indices of the filters 
		you are trying to visualize. I will, at some point, make these user argument
		...I think.
	3. Most interesting results come from running on either the final fully connected
		(dense) layer or the 2nd (non-input) convolutional layer.
	4. The names of the layers can be seen by running the script once, it will 
		output a text representation of the model that includes dimensions
		of output and input as well as layer names and types.
'''

#pick which layer you would like to run visualize on
#TODO: write separate script that will run over multiple layers
layer_name = 'convolution2d_2' # TODO: don't forget to change the loss function
#dense_2 is the name of the final fully-connected classification layer
#convolution2d_2 is the name of the second non-input convolution layer

img_width = 32
img_height = 32


#this function takes in an image and processes it into a usable grayscale image
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

#load model and weights (generated in mnist_cnn.py)
model = load_model('../hareesh/model_32x32_35k.h5')
print('loaded model')
model.load_weights('../hareesh/weights_32x32_35k.h5')
#have to load weights manually to exclude top layers
print('loaded weights')
#print model summary
model.summary()

#define the step for gradient ascent
step = 0.5

#create a dictionary with each layer's name for convenience
layer_dict = dict([(layer.name, layer) for layer in model.layers])


#start with initial input, obviously
#print(model.layers[0].input)
input_img = model.layers[0].input
	#Note: the layer index would be -1 instead of 0 but this network uses the first 
	#convolutional layer as the input instead of a fully-conected

layer_output = layer_dict[layer_name].output
#layer_output = model.layers[0].output
for n in range(0,5):
	filter_index = n
	print('Processing filter %d' % filter_index)
	start_time = time.time()
	#build a loss function that maximizes the activation of n filters of the layer considered

	#TODO: select one of these loss function depending on which layer you're using
	#loss = backend.mean(layer_output[:, filter_index]) # loss function for dense layers
	loss = backend.mean(layer_output[:,filter_index,:,:]) # for non-dense layers

	#compute gradient of input picture wrt this loss
	grads = backend.gradients(loss, input_img)[0]
	#normalize gradient to avoid values
	grads /= (backend.sqrt(backend.mean(backend.square(grads))) + 1e-5)

	#this function returns loss and grads given the input picture
	iterate = backend.function([input_img, backend.learning_phase()],[loss,grads])

	#set numpy's random seed for reproducibility
	numpy.random.seed(117)

	#start with random grayscale image to begin gradient ascent on
	input_img_data = numpy.random.random((1,5,img_width, img_height))*20+128

	#run gradient ascent on current filter for x steps until loss function is maximized
	for i in range(1000):
		#compute loss and gradients
		loss_value, grads_value = iterate([input_img_data, 0]) # 2nd argument is always 0 (for test phase)
		#apply gradient to image and repeat
		input_img_data += grads_value * step
	
		print('Current loss value: %d'% loss_value)

	#convert array to grayscale image
	img = deprocess_image(input_img_data[0])
	end_time = time.time()
	print('Filter %d processed in %ds' % (filter_index, end_time-start_time))
	#save image to file
	scipy.misc.toimage(img[:,:,0]).save('./images/tt_cnn_layer_%s_filter_%d.png' %(layer_name, filter_index))

