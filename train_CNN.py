#Copyright [2017] [Alexej Berger <alexej.berger@t-online.de>]

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#http://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

#The above copyright notice and this permission notice shall be
#included in all copies or substantial portions of the Software.

import os
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import time

# Set parameters
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

classes_amount = 4

# used to rescale the pixel values from [0, 255] to [0, 1] interval
datagen = ImageDataGenerator(
        rescale=1./255,        # normalize pixel values to [0,1]
        shear_range=0.2,       # randomly applies shearing transformation
        zoom_range=0.2,        # randomly applies shearing transformation
        horizontal_flip=True)  # randomly flip the images

# img = load_img('data/train/class1/img_1.jpg')
# x = img_to_array(img)
# x = x.reshape((1,) + x.shape)
# i = 0
# for batch in datagen.flow(x, batch_size=1,
#                           save_to_dir='preview', save_prefix='class1', save_format='jpeg'):
#     i += 1
#     if i > 20:
#         break  # otherwise the generator would loop indefinitely

# automagically retrieve images and their classes for train and validation sets
train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=16,
        class_mode='categorical')

validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=16,
        class_mode='categorical')

#Model
model = Sequential()
# Convlutional Layer, 32 filters with a 3x3 Convlutional
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))   #32
#relu for nonlinearities
model.add(Activation('relu'))
#Reducing the complexity of feature maps. 2x2 filter
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))       #32
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))      #64
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#preventing overfitting
#Flatten into 1D
model.add(Flatten())
#Fully connected layer
model.add(Dense(64))               #64
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(int(classes_amount), activation='softmax'))
#model.add(Dense(1))
#model.add(Activation('sigmoid'))    #only for binary classes

# categorical_crossentropy for more that 2 classes. binary_crossentropy otherwise
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',              #rmsprop
              metrics=['accuracy'])

batch_size = 16
nb_epoch = 30
nb_train_samples = 232*classes_amount                # old_data == 283,data ==214
nb_validation_samples = 29*classes_amount            # old_data == 38,data ==27

model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples / batch_size,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples / batch_size)

# Save Model
model_json = model.to_json()
with open("models/basic_cnn_30_epochs_data.json", "w") as json_file:
    json_file.write(model_json)

#Save Weights
model.save_weights('models/basic_cnn_30_epochs_data.h5')

##### TEST

time_ = time.clock()
test_img = load_img('some_pics/640_leo_dicaprio_emma_watson.jpg', target_size=(img_width,img_height))
#test_img = load_img('data/validation/Emma_Watson/pic_294.jpg', target_size=(200,200))
test_img.show()
image_as_array = img_to_array(test_img)
image_as_array = image_as_array.reshape((1,) + image_as_array.shape)
prediction = model.predict(image_as_array)              # for vector output
#prediction = model.predict_classes(image_as_array)      # for classes output
print ("Time:%.4f" %(time.clock()-time_))
print prediction
