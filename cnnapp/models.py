from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import InputLayer, Dense, Activation, Dropout, Flatten
 
from django.db import models
import numpy as np
from PIL import Image
import io, base64
 
 
class Pict(models.Model):
  image = models.ImageField(upload_to='picture')
 
  IMAGE_SIZE = 32
  MODEL_FILE_PATH = './carbike/trainedmodel/model_cnn_cifer.h5'
  CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
  NUM_CLASSES = len(CLASSES)
 
  def predict(self):
 
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
 
    model.add(Dense(self.NUM_CLASSES, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
    model.summary()
 
    # predict
    img_data = self.image.read()
    img_bin = io.BytesIO(img_data)
    image = Image.open(img_bin)
    image = image.convert("RGB")
    image = image.resize((self.IMAGE_SIZE, self.IMAGE_SIZE))
    data = np.asarray(image) / 255.0
    X = []
    X.append(data)
    X = np.array(X)
 
    result = model.predict([X])[0]
    predicted = result.argmax()
    rate = int(result[predicted] * 100)
 
    print(self.CLASSES[predicted], rate)
    return self.CLASSES[predicted], rate
 
 
  def image_src(self):
    with self.image.open() as img:
      base64_img = base64.b64encode(img.read()).decode()
 
      return 'data:' + img.file.content_type + ';base64,' + base64_img