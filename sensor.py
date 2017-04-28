import RPi.GPIO as GPIO
import time
import os
 
#Chosen GPIO
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(26, GPIO.IN)
 
def movement(channel):
    # Start pipeline
    print('Movement Detected!')
 
try:
    GPIO.add_event_detect(26, GPIO.RISING, callback=movement)
    print 
    while True:
        time.sleep(100)
        
except KeyboardInterrupt:
    GPIO.cleanup()
    print "Closing!"

