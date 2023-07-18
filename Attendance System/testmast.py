import cv2
import time
import face_recognition
import numpy as np
import csv
import os
from datetime import datetime
import argparse
import warnings
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

SAMPLE_IMAGE_PATH = "./images/sample/"

video_capture = cv2.VideoCapture(0)

sai_sreekar_image = face_recognition.load_image_file("images/sample/Sai Sreekar.jpg")
sai_sreekar_encoding = face_recognition.face_encodings(sai_sreekar_image)[0]

harika_image = face_recognition.load_image_file("images/sample/har.jpeg")
harika_encoding = face_recognition.face_encodings(harika_image)[0]

sathvika_image = face_recognition.load_image_file("images/sample/sat.jpeg")
sathvika_encoding = face_recognition.face_encodings(sathvika_image)[0]

siri_image = face_recognition.load_image_file("images/sample/siri.jpg")
siri_encoding = face_recognition.face_encodings(siri_image)[0]

soma_image = face_recognition.load_image_file("images/sample/Soma.jpg")
soma_encoding = face_recognition.face_encodings(soma_image)[0]
known_face_encoding = [sai_sreekar_encoding,harika_encoding,sathvika_encoding,siri_encoding,soma_encoding]

known_faces_names = ["620217","620112","620113","620228","620255"]
s=True
def recog():
    while True:
        _,frame = video_capture.read()
        small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
        rgb_small_frame = small_frame[:,:,::-1]
        if s:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
                name=""
                face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
                best_match_index = np.argmin(face_distance)
                if matches[best_match_index]:
                    name = known_faces_names[best_match_index]

        cv2.imshow("Attendence System",frame)
        if cv2.waitKey(1) & 0xFF == ord('r'):
            break

    return name

def cap(name):   
# Open the camera
    cap = cv2.VideoCapture(0)
   
  
    while True:
      
    # Read and display each frame
        ret, img = cap.read()
        cv2.imshow(name, img)
  
    # check for the key pressed
        k = cv2.waitKey(125)
  
    # set the key for the countdown
    # to begin. Here we set q
    # if key pressed is q
        if k == ord('c'):
  
            ret, img = cap.read()

            # Save the frame
            cv2.imwrite(name+'.jpg', img)

            rot=cv2.imread(name+'.jpg')

            image=cv2.rotate(rot,cv2.ROTATE_90_CLOCKWISE)

            cv2.imshow(name, image)

            cv2.waitKey(1000)

            cv2.imwrite(name+'.jpg', image)
  
            # HERE we can reset the Countdown timer
            # if we want more Capture without closing
            # the camera
  
    # Press Esc to exit
        elif k == 27:
            break
  
# close the camera
    cap.release()
   
# close all the opened windows
    cv2.destroyAllWindows()

cap(recog())
