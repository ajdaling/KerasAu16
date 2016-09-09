from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils


(x_train, y_train), (x_test, y_test) = mnist.load_data()

plt.subplot(221)
plt.imshow(x_train[0], cmap=plt.get_cmap('gray'))
plt.show()

#fix random seed
seed = 7
numpy.random.seed(seed)

#reshape images to pixel vectors (28x28 to 784)
num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')

#normalize pixels from 0-255 (grayscale) to 0-1
x_train = x_train / 255
x_test = x_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

#define baseline model
def baseline_model():
    #create model
    model = Sequential()
    #simple neural net w/ one hidden layer with same # neurons as inputs (784)
    #rectifier activation function
    
    model.add(Dense(num_pixels, input_dim = num_pixels, init='normal', activation='relu'))
    
    #softmax activation function on output
    #adam gradient descent
    # logarithmic loss (crossentropy)
    model.add(Dense(num_classes, init='normal',activation='softmax'))
    #Compile Model
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

#build model
model = baseline_model()
#fit model
model.fit(x_train,y_train, validation_data=(x_test,y_test), nb_epoch = 10, batch_size = 200, verbose = 2)
#final evalutation
scores = model.evaluate(x_train, y_train, verbose=0)
print("Baseline error: %.2f%%" % (100-scores[1]*100))