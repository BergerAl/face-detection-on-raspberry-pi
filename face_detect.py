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
import time
#import numpy as np

#transmitting faces to the neural network
def identification(face_image):
    #load weights

    #identificate the face_image
    aaaa =20
time_ = time.clock()
# python face_detect.py IMG_20170307_170844.jpg haarcascade_frontalface_default.xml
# Get user supplied values
imagePath = sys.argv[1]

#cascade for detection
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#faceCascade = cv2.CascadeClassifier('/home/pi/Desktop/face_recog/lbpcascade_frontalface.xml')

# Read the image and resize it
image = cv2.imread(imagePath)

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

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    print (x, y, w, h)
    # Make new image for detection
    recog = image[y-h/2:y+h*3/2, x-w/2:x+w*3/2]
    cv2.imshow("small", recog)      #here you see the image for the CNN
    cv2.imwrite('test.jpg', recog)

    #Detection in whole image
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)


cv2.imshow("Faces found", image)
print (time.clock()- time_)
cv2.waitKey(0)
