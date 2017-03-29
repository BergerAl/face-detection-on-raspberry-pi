import os
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator

# Set parameters
img_width, img_height = 100, 100

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
        batch_size=32,
        class_mode='categorical')

#Model
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(img_width, img_height, 3)))   #32
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))       #32
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))      #64
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))               #64
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(classes_amount, activation='softmax'))
#model.add(Dense(1))
#model.add(Activation('sigmoid'))    #only for binary classes


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

batch_size = 16
nb_epoch = 30
nb_train_samples = 214*4                # old_data == 283,data ==214
nb_validation_samples = 27*4            # old_data == 38,data ==27

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
