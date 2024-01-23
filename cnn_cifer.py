import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import InputLayer, Dense, Activation, Dropout, Flatten
from tensorflow.keras.utils import plot_model, to_categorical
from keras.callbacks import TensorBoard
from keras.datasets import cifar10
from numpy import argmax
 
EPOCHS = 10
BATCH_SIZE = 128
CIFER10_CLASSES = 10
LABEL_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
MODELFILE = 'model_cnn_cifer.h5'
PREDICT_NUM = 5
 
(images_train, labels_train),(images_test, labels_test) = cifar10.load_data()
 
images_train = images_train.astype('float32')
images_test = images_test.astype('float32')
images_train /= 255.0
images_test /= 255.0
 
# one-hot
labels_train = to_categorical(labels_train, CIFER10_CLASSES)
labels_test = to_categorical(labels_test, CIFER10_CLASSES)
 
# create model
model = Sequential()
model.add(InputLayer(input_shape=(32,32,3)))
 
model.add(Conv2D(32,3))
model.add(Activation('relu'))
 
model.add(Conv2D(32,3))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
 
model.add(Conv2D(64,3))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
 
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
 
model.add(Dropout(0.5))
 
model.add(Dense(CIFER10_CLASSES, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
model.summary()
 
# train
if not os.path.exists(MODELFILE):
  history = model.fit(images_train, labels_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_split=0.1)
  print('Train done.')
 
  model.save_weights(MODELFILE)
  print('Train saved.')
 
else:
  # load model
  model.load_weights(MODELFILE)
  print('Train loaded.')  
 
# predict
predicted = model.predict(images_test)
for n in range(PREDICT_NUM):
  print('predict = ', LABEL_NAMES[argmax(predicted[n])])
  print('label = ', LABEL_NAMES[argmax(labels_test[n])])
  print('')