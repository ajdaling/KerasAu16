from keras.datasets import mnist
import maplotlib.pyplot as plt
import numpy
from keras.models import load_model
from keras.models import Sequential
from keras.layers import dense
from keras.layers import Dropout
from keras.utils import np_utils
import theano

model = load_model('mnist_model1.h5')

img_width = 28
img_height = 28

#name of layer to visualize
layer_name = 'dense2'


#util function to convert a tensor into an image
def deprocess_image(x):
	#normalize tensor: cneter on 0
	x -= x.mean()
	x /= (x.std() + 1e-5)
	x *= 0.1
	
	#clip to [0,1]
	x += 0.5
	x = np.clip[x,0,1]

	#convert to grayscale
	
#TODO: figure out how to do this



#load model and weights
json_file = open('mnist_model2.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
print('loaded model')
model.load_weights('mnist_Weights.h5')
print('loaded weights')

#print model summary
model.summary()

#arbitrarily select filter 10
filter_index = 10

print('Processing filter %d' % filter_index)
start_time = time.time()


	










