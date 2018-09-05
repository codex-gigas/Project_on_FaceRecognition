import cv2,os
import numpy as np
from PIL import Image

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
