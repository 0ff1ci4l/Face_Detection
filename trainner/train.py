import os
import pickle
import numpy as np
from cv2 import cv2
from PIL import Image


dir_ = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(dir_,"faces")

cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_id = {}

x_train = []
y_label = []

for root,dirs,files in os.walk(img_dir):
    for file in files:
        if file.endswith("jpg"):
            path = os.path.join(root,file)
            label = os.path.basename(root).replace(" ", "_").lower()
            print(label,path)
            if label in label_id:
                pass
            else:
                label_id[label] = current_id
                current_id += 1
            _id = label_id[label]
            print(label_id)

            pillow_img = Image.open(path).convert("L")
            img_array = np.array(pillow_img,"uint8")
            print(img_array)
            faces = cascade.detectMultiScale(img_array,scaleFactor=1.3,minNeighbors=5)

            for (x,y,w,h) in faces:
                roi = img_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_label.append(_id)

with open("labels.pickle",'wb') as f:
    pickle.dump(label_id,f)

recognizer.train(x_train,np.array(y_label))
recognizer.save("trainner.yml")