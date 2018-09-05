import cv2,os
import numpy as np
import sqlite3
from PIL import Image
import pickle
path = 'T:/PyProjects/projects/facial_project/ImageData'
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
path = 'T:/PyProjects/projects/facial_project/ImageData'
cam=cv2.VideoCapture(0)
recognizer = cv2.createLBPHFaceRecognizer()
recognizer.load('T:\\PyProjects\\projects\\facial_project\\face_recognizer\\trained_model.yml')
Img_Path = [os.path.join(path,i)for i in os.listdir(path)]
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,2,1,0,2)
while(True):
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.25,5,minSize=(100,100),flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        id,conf=recognizer.predict(gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),1)
        for image_path in Img_Path:
            Id = int(os.path.split(image_path)[-1].split('_')[1])
            if(id==Id):
                id=os.path.split(image_path)[-1].split('_')[0]
##        if(id==3):
##            id='nyamath'
                cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h+30),font,255)
    cv2.imshow('Detecting the Face..',img)
    if(cv2.waitKey(1)==ord('q')):
       break
cam.release()
cv2.destroyAllWindows()
