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
#included in all copies or substantial portions of the Software

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import RPi.GPIO as GPIO
import time
import picamera
import cv2
import sys
from numpy import arange
from keras.models import model_from_json
from keras.preprocessing.image import array_to_img, img_to_array, load_img

#Input with Button and Output an LED Signal
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(21, GPIO.IN, pull_up_down=GPIO.PUD_UP)       #Switch
GPIO.setup(16, GPIO.OUT)                                #Green LED
GPIO.setup(20, GPIO.OUT)                                #Red LED
GPIO.setup(12, GPIO.OUT)                                #Yellow LED
GPIO.setup(26, GPIO.IN)                                 #Movement Detector Pin                                   


# Camera
camera= picamera.PiCamera()
camera.resolution = (1000,1000)

class face_detect():
    def __init__(self, runtime_after_detection):
        self.runtime_after_detection = runtime_after_detection
        self.model_done = False

    def name_assignment(self, prediction_array):
        person_name = 'Not a known person'
        name_list = ['Anne Hathaway', 'Emma Watson', 'Leonardo Dicaprio', 'Tom Hardy']

        for i in arange(len(name_list)):
            if prediction_array[0,i]>=0.9:
                person_name = name_list[i]
            else:
                pass
        return person_name

    def sending_on_phone(self, image_name, person_list):
        from smtplib         import SMTP_SSL
        from email.mime.image import MIMEImage
        from email.mime.multipart import MIMEMultipart
        
        #logindata
        login, password = 'sender_email@gmx.de', 'password'
        recipients = ['recipient_email']

        #Msg
        msg = MIMEMultipart()

        msg['Subject'] = '%s is at the door' %(", ".join(person_list))
        msg['From'] = 'sender_email@gmx.de'
        msg['To'] = ", ".join(recipients)


        fp = open(image_name, 'rb')
        face = MIMEImage(fp.read())
        fp.close()
        msg.attach(face)

        #send it via gmx smtp
        s = SMTP_SSL('mail.gmx.net', 465, timeout=10)
        s.set_debuglevel(1)
        try:
            s.login(login, password)
            s.sendmail(msg['From'], recipients, msg.as_string())
        finally:
            s.quit()
        
    def load_libs_and_model(self):
        

        #Face recognition cascade
        self.faceCascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

        #load model and weights
        json_file = open('models/basic_cnn_30_epochs_data.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)

        self.model.load_weights("models/basic_cnn_30_epochs_data.h5")
        self.model_done = True
        print ("Everything was loaded as intended")

    def run_detection(self):
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
            image_name = 'python_pictures/' + date + '.jpg'
            image = cv2.imread(image_name)
            r = 600.0/image.shape[1]
            dim = (600, int(image.shape[0]*r))
            image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = self.faceCascade.detectMultiScale(
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

            else:
                person_list = []
                for (x, y, w, h) in faces:
                    
                    boundary_factor = 0.1
                    # Extracting face from image
                    recog = image[y-int(boundary_factor*h):y+int(h*(1+boundary_factor)), x-int(boundary_factor*w):x+int(w*(1+boundary_factor))]
                    ### Analysing in CNN ###
                    recog = cv2.resize(recog, (150,150))
                    image_as_array = img_to_array(recog)
                    image_as_array = image_as_array.reshape((1,) + image_as_array.shape)
                    #need a bigger array or something
                    prediction = self.model.predict(image_as_array)
                    print prediction
                    person_list.append(self.name_assignment(prediction))
                    
                print person_list
                self.sending_on_phone(image_name, person_list)
                GPIO.output(12, 0)
                GPIO.output(16, 1)
                person_list = []
                
                time.sleep(5)
                GPIO.output(16, 0)
                


    def main(self, channel):
        print "Movement Detected! Starting Algorithm"
        if self.model_done == False:
            self.load_libs_and_model()
        else:
            pass
        start_time = time.time()
        while (time.time()-start_time < self.runtime_after_detection):
            GPIO.output(12, 0)
            self.run_detection()

            
FD = face_detect(30)
GPIO.output(12, 0)
try:
    GPIO.add_event_detect(26, GPIO.RISING, callback=FD.main)
    while True:
        time.sleep(100)
     
       
except KeyboardInterrupt:
    GPIO.cleanup()
    print "Closing!"


 
