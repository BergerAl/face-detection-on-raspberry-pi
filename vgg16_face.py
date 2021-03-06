# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# download vgg16 weights at https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

import os
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.utils.np_utils import to_categorical
from keras import optimizers
import h5py

# images
img_width, img_height = 150, 150
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
classes_amount = 4
batch_size = 16

# used to rescale the pixel values from [0, 255] to [0, 1] interval
datagen = ImageDataGenerator(rescale=1./255)

# Architecture vgg16
model_vgg = Sequential()
model_vgg.add(ZeroPadding2D((1, 1), input_shape=(img_width, img_height,3)))
model_vgg.add(Conv2D(64, (3, 3), activation='relu', name='conv1_1'))
model_vgg.add(ZeroPadding2D((1, 1)))
model_vgg.add(Conv2D(64, (3, 3), activation='relu', name='conv1_2'))
model_vgg.add(MaxPooling2D((2, 2), strides=(2, 2)))

model_vgg.add(ZeroPadding2D((1, 1)))
model_vgg.add(Conv2D(128, (3, 3), activation='relu', name='conv2_1'))
model_vgg.add(ZeroPadding2D((1, 1)))
model_vgg.add(Conv2D(128, (3, 3), activation='relu', name='conv2_2'))
model_vgg.add(MaxPooling2D((2, 2), strides=(2, 2)))

model_vgg.add(ZeroPadding2D((1, 1)))
model_vgg.add(Conv2D(256, (3, 3), activation='relu', name='conv3_1'))
model_vgg.add(ZeroPadding2D((1, 1)))
model_vgg.add(Conv2D(256, (3, 3), activation='relu', name='conv3_2'))
model_vgg.add(ZeroPadding2D((1, 1)))
model_vgg.add(Conv2D(256, (3, 3), activation='relu', name='conv3_3'))
model_vgg.add(MaxPooling2D((2, 2), strides=(2, 2)))

model_vgg.add(ZeroPadding2D((1, 1)))
model_vgg.add(Conv2D(512, (3, 3), activation='relu', name='conv4_1'))
model_vgg.add(ZeroPadding2D((1, 1)))
model_vgg.add(Conv2D(512, (3, 3), activation='relu', name='conv4_2'))
model_vgg.add(ZeroPadding2D((1, 1)))
model_vgg.add(Conv2D(512, (3, 3), activation='relu', name='conv4_3'))
model_vgg.add(MaxPooling2D((2, 2), strides=(2, 2)))

model_vgg.add(ZeroPadding2D((1, 1)))
model_vgg.add(Conv2D(512, (3, 3), activation='relu', name='conv5_1'))
model_vgg.add(ZeroPadding2D((1, 1)))
model_vgg.add(Conv2D(512, (3, 3), activation='relu', name='conv5_2'))
model_vgg.add(ZeroPadding2D((1, 1)))
model_vgg.add(Conv2D(512, (3, 3), activation='relu', name='conv5_3'))
model_vgg.add(MaxPooling2D((2, 2), strides=(2, 2)))

# break the layers appart. We need only the first few

f = h5py.File('models/vgg16_weights.h5')
for k in range(f.attrs['nb_layers']):
    if k >= len(model_vgg.layers) - 1:
        # we don't look at the last two layers in the savefile (fully-connected and activation)
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    layer = model_vgg.layers[k]

    if layer.__class__.__name__ in ['Conv1D', 'Conv2D', 'Conv3D', 'AtrousConv2D']:
        weights[0] = np.transpose(weights[0], (2, 3, 1, 0)) #2,3,1,0

    layer.set_weights(weights)

f.close()

#bottleneck

nb_train_samples = 232*classes_amount               #x*batch_size
nb_validation_samples = 28*classes_amount            #y*batch_size

train_generator_bottleneck = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

validation_generator_bottleneck = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)


# Comment this part
bottleneck_features_train = model_vgg.predict_generator(train_generator_bottleneck, nb_train_samples // batch_size)
np.save(open('models/bottleneck_features_train.npy', 'wb'), bottleneck_features_train)

bottleneck_features_validation = model_vgg.predict_generator(validation_generator_bottleneck, nb_validation_samples // batch_size)
np.save(open('models/bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)

# load the output
train_data = np.load(open('models/bottleneck_features_train.npy', 'rb'))
train_labels = np.array([0] * (nb_train_samples / classes_amount) + [1] * (nb_train_samples / classes_amount) + [2] * (nb_train_samples / classes_amount) + [3] * (nb_train_samples / classes_amount))


validation_data = np.load(open('models/bottleneck_features_validation.npy', 'rb'))
validation_labels = np.array([0] * (nb_validation_samples / classes_amount) + [1] * (nb_validation_samples / classes_amount) + [2] * (nb_validation_samples / classes_amount) + [3] * (nb_validation_samples / classes_amount))

#Transform both outputs to categorical, so the output is a binarical matrix
train_labels = to_categorical(train_labels, num_classes=classes_amount)
validation_labels = to_categorical(validation_labels, num_classes=classes_amount)

# Add new layers to the end of VGG16 for recognition
model_top = Sequential()
model_top.add(Flatten(input_shape=train_data.shape[1:]))
model_top.add(Dense(256, activation='relu'))
model_top.add(Dropout(0.5))
model_top.add(Dense(int(classes_amount), activation='softmax'))
#model_top.add(Dense(1, activation='sigmoid'))

model_top.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
nb_epoch=40
model_top.fit(train_data, train_labels,
          epochs=nb_epoch, batch_size=batch_size,
          validation_data=(validation_data, validation_labels))

# Save new Weights
model_top.save_weights('models/bottleneck_face_40_epochs.h5')
print "Training done!"
# Evaluation

# Save Model
model_json = model_top.to_json()
with open("models/bottleneck_face_40_epochs_model.json", "w") as json_file:
    json_file.write(model_json)
