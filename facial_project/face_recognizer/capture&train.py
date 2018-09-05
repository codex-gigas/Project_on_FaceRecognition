import cv2,time,os
import numpy as np
from PIL import Image
import sqlite3

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0)
id = raw_input('Enter a number that would be your ID: ')
name = raw_input('Enter your name: ')
sample = 0
while(True):
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        sample+=1
        cv2.rectangle(img,(x-50,y-50),(x+w,y+h),(0,255,0),1) 
        cv2.imshow('Capturing image... ',img) 
        cv2.imwrite('T:/PyProjects/projects/facial_project/ImageData/'+str(name) + "_" + str(id) + '_' + str(sample) + '.jpg',gray[y:y+h,x:x+w])
    cv2.waitKey(1)     
    if sample>50:
        break
cam.release()
cv2.destroyAllWindows()

print('Please wait! training the model...')
time.sleep(2)
recog = cv2.createLBPHFaceRecognizer()
detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

path = 'T:/PyProjects/projects/facial_project/ImageData'
def Model_Training(path):
    Img_Path = [os.path.join(path,i)for i in os.listdir(path)]
##    print(Img_Path)
    Samples = []
    Ids = []
    for image_path in Img_Path:
        if(os.path.split(image_path)[-1].split(".")[-1]!='jpg'):
            continue
        Img = Image.open(image_path).convert('L')
        np_array = np.array(Img,'uint8')
        Id = int(os.path.split(image_path)[-1].split('_')[1])
        faces = detect.detectMultiScale(np_array)
        for(x,y,w,h) in faces:
            Samples.append(np_array[y:y+h,x:x+w])
            Ids.append(Id)
    return Samples,Ids

faces,Ids = Model_Training(path)
recog.train(faces,np.array(Ids))
recog.save('trained_model.yml')
print('The Model is trained sucessfully!!')



