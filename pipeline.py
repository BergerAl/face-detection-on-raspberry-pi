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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import sys
from keras.models import model_from_json
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import RPi.GPIO as GPIO
import time
import picamera

#Face recognition cascade
faceCascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')



#load model and weights
json_file = open('models/basic_cnn_30_epochs_data.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights("models/basic_cnn_30_epochs_data.h5")

#Input with Button and Output an LED Signal
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(21, GPIO.IN, pull_up_down=GPIO.PUD_UP)       #Switch
GPIO.setup(16, GPIO.OUT)                                #Green LED
GPIO.setup(20, GPIO.OUT)                                #Red LED
GPIO.setup(12, GPIO.OUT)                                #Yellow LED

# Camera
camera= picamera.PiCamera()
camera.resolution = (1000,1000)

print ("Everything was loaded as intended")

while True:
    input_state = GPIO.input(21)
    if input_state == False:
        GPIO.output(12, 1)
        date = time.strftime("%Y-%m-%d_%H:%M:%S")

        #camera.start_preview()
        time.sleep(1)
        camera.capture('python_pictures/' + date + '.jpg')
        print ("Picture captured")

        ### Start processing ###
        # Read the image and resize it
        image = cv2.imread('python_pictures/' + date + '.jpg')
        r = 600.0/image.shape[1]
        dim = (600, int(image.shape[0]*r))
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        if faces == ():
            GPIO.output(20, 1)
            print ("No faces found")
            time.sleep(3)
            GPIO.output(20, 0)
            continue
        else:
            for (x, y, w, h) in faces:
                boundary_factor = 0.1
                # Make new image for detection
                recog = image[y-int(boundary_factor*h):y+int(h*(1+boundary_factor)), x-int(boundary_factor*w):x+int(w*(1+boundary_factor))]

        ### Analysing in CNN ###
        recog = cv2.resize(recog, (150,150))
        image_as_array = img_to_array(recog)
        image_as_array = image_as_array.reshape((1,) + image_as_array.shape)
        prediction = model.predict(image_as_array)
        GPIO.output(12, 0)
        GPIO.output(16, 1)
        print prediction
        time.sleep(5)
        GPIO.output(16, 0)
