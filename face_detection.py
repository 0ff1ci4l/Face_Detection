import random
import pickle
from cv2 import cv2

cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
labels = {}

with open("labels.pickle",'rb') as f:
    original_labels = pickle.load(f)
    labels = {v:k for k,v in original_labels.items()}

image1 = cv2.imread("jack.jpg")
image2 = cv2.imread("elon.jpg")
image3 = cv2.imread("jeff.jpg")
image4 = cv2.imread("beast.jpg")

random_face = [image1,image2,image3,image4]
random_face = random.choice(random_face)

while True: 
    gray = cv2.cvtColor(random_face,cv2.COLOR_BGR2GRAY)
    face = cascade.detectMultiScale(gray, scaleFactor=1.3,minNeighbors=5)
    
    for (x,y,h,w) in face:
        cv2.rectangle(random_face,(x,y),(x+w,y+h),(255,0,0),2)
        r_gray = gray[y:y+h, x:x+w]
        r_color = random_face[y:y+h, x:x+w]

        _id,config = recognizer.predict(r_gray)
        if config >= 45 and config <= 85:
            pass
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            names = labels[_id]
            color = (255,255,255)
            stroke = 2
            cv2.putText(random_face,names, (x,y), font, 1, color, stroke, cv2.LINE_AA)
    
    cv2.imshow('did i guessed that correctly',random_face)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()