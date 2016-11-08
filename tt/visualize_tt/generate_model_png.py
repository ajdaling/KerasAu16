import numpy
import scipy
from keras.models import load_model
from keras.utils import np_utils
from keras.utils.visualize_util import plot
from keras import backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
import scipy
import theano
import time
import h5py



model = load_model('../hareesh/model_32x32_35k.h5')
plot(model, to_file = "model_32x32_35k.png")
