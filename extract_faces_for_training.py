#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv3
import sys
import time
import os

imagePath = 'data/Emma_Watson'

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

for filename in os.listdir(imagePath):

    image = cv2.imread(imagePath + '/' + filename)
    cv2.imshow("pic", image)
    cv2.waitKey(0)
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
        cv2.imshow("small", recog)      #here you see the image for the CNN

        #Detection in whole image
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.waitKey(0)
