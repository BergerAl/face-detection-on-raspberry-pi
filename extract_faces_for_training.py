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
import os

Person = 'Anne_Hathaway'
imagePath = 'data/validation/' + Person +'/'

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if not os.path.exists(Person + "/"):
    os.makedirs(Person + "/")
    print "New folder generated"

for filename in os.listdir(imagePath):

    image = cv2.imread(imagePath + '/' + filename)
    #cv2.imshow("pic", image)
    #cv2.waitKey(0)
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
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        # Make new image for detection

        recog = image[y:y+h, x:x+w]
        #cv2.imshow("small", recog)      #here you see the image for the CNN
        cv2.imwrite(Person + "/" + filename, recog)

        #Detection in whole image
        #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #cv2.imshow("Faces found", image)

    #cv2.waitKey(0)
