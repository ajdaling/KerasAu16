import keras
import numpy
import pandas

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from keras import backend as K



print 'Loading data'

img_width, img_height = 32, 32
numChannels=5

train_data_dir = 'data/train'
test_data_dir = 'data/test'
#nb_train_samples = 2000
#nb_validation_samples = 800
nb_epoch = 5
batch_size = 15
nevents = 35000
nlines = nevents*5

dataset = pandas.read_csv("/nfs/09/osu9669/work/particleFlow/TorchData/ttHbb_Btag/xaa", delimiter=",", header=None,dtype='a')
print dataset.values.size
data_sig = dataset.values[:,0:(img_width*img_height)].astype(float)
labels_sig = dataset.values[:,(img_width*img_height)].astype(float)
runNum_sig = dataset.values[:,(img_width*img_height+1)].astype(float)
nevents_sig = (data_sig.shape[0])/numChannels

print data_sig.shape
print labels_sig.shape
print nevents_sig

if numChannels > 1:
	for i in range(0,nevents_sig):
		if runNum_sig[i*numChannels] != runNum_sig[i*numChannels+1]:
			print 'run mismatch!: ',runNum_sig[i*numChannels],"; ",runNum_sig[i*numChannels+1]
		if labels_sig[i*numChannels] != labels_sig[i*numChannels+1]:
			print 'run mismatch!: ',labels_sig[i*numChannels],"; ",labels_sig[i*numChannels+1]


#data_sig=numpy.reshape(data_sig,(img_width,img_height,numChannels,nevents_sig))
#data = numpy.resize(data,(img_width,img_height,numChannels,nevents))
#labels_sig = labels_sig[::5]

dataset = pandas.read_csv("/nfs/09/osu9669/work/particleFlow/TorchData/ttjets_Btag/xaa", delimiter=",", header=None,dtype='a')
data_back = dataset.values[:,0:(img_width*img_height)].astype(float)
labels_back = dataset.values[:,(img_width*img_height)].astype(float)
runNum_back = dataset.values[:,(img_width*img_height+1)].astype(float)
nevents_back = (data_back.shape[0])/numChannels


if numChannels > 1:
	for i in range(0,nevents_back):
		if runNum_back[i*numChannels] != runNum_back[i*numChannels+1]:
			print 'run mismatch!: ',runNum_back[i*numChannels],"; ",runNum_back[i*numChannels+1]
		if labels_back[i*numChannels] != labels_back[i*numChannels+1]:
			print 'run mismatch!: ',labels_back[i*numChannels],"; ",labels_back[i*numChannels+1]


#data_back=numpy.reshape(data_back,(img_width,img_height,numChannels,nevents_back))
#data = numpy.resize(data,(img_width,img_height,numChannels,nevents))
#labels_back = labels_back[::5]
print labels_back.shape

labels_all=numpy.concatenate((labels_sig,labels_back),axis=0)
data_all=numpy.concatenate((data_sig,data_back),axis=0)

print data_all.shape
print labels_all.shape

#data_all=numpy.reshape(data_a,(img_width,img_height,numChannels,nevents_back))
labels_all = labels_all[::5]
labels_all = labels_all-1

shuffler = numpy.random.permutation(labels_all.shape[0])
portionTrain = 0.8
trsize = numpy.floor(shuffler.size*portionTrain)
tesize = shuffler.size - trsize

X_train = data_all[0:(trsize*5),:]
y_train = labels_all[0:trsize]

X_test = data_all[trsize*5:data_all.shape[0],:]
y_test = labels_all[trsize:labels_all.shape[0]]
print X_train.shape
#print y_train.shape
print X_test.shape
#print y_test.shape

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(y_train.shape[0], 5, img_height, img_width)
    X_test = X_test.reshape(y_test.shape[0], 5, img_height, img_width)
    input_shape = (5, img_height, img_width)
else:
    X_train = X_train.reshape(y_train.shape[0], img_height, img_width, 5)
    X_test = X_test.reshape(y_test.shape[0], img_height, img_width, 5)
    input_shape = (img_height, img_width, 5)
    

Y_train = np_utils.to_categorical(y_train, 2)
Y_test = np_utils.to_categorical(y_test, 2)
    
    
model = Sequential()
model.add(Convolution2D(10, 5, 5, input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(5, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(10))
model.add(Activation('linear'))
model.add(Dense(2))
model.add(Activation('softmax'))
#model.add(Dense(1))
#model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])


model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

#save model to file
model.save('model_32x32_11886.h5')
print('saved model once')

model_json = model.to_json()
with open('model_32x32_11886.json','w+') as json_file:
	json_file.write(model_json)
print('saved model twice')
#save weights to file
model.save_weights('weights_32x32_11886.h5')
print('saved weights')

print 'done'
