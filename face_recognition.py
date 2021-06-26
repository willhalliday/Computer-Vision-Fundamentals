import numpy as np
import cv2 as cv
import faces_train
haar_cascade = cv.CascadeClassifier('C:/Users/Will/Downloads/Python/Computer_Vision/haar.face.xml')

people = ['Ben Affleck', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling', 'Mitch Robinson']
# features = np.load('features.npy', allow_pickle=True)
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yaml')

img = cv.imread(r'C:/Users/Will/Downloads/Python/Computer_Vision/Faces/val/ben_affleck/3.jpg')
img = cv.resize(img, (618, 870))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, 2, 4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    cv.putText(img, "DETECTED: " + str(people[label]) + " (Confidence = " + str(confidence) +")", (10,80), cv.FONT_HERSHEY_COMPLEX, 0.60, (0,0, 255), thickness=2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,0,255), thickness=3)

cv.imshow('Detected Face', img)

cv.waitKey(0)

