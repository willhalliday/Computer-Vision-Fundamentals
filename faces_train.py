import os
import cv2 as cv
import numpy as np

people = ['Ben Affleck', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling', 'Mitch Robinson']
DIR = r'C:/Users/Will/Downloads/Python/Computer_Vision/Faces/train'

# Establish classifier and empty feature/label lists
haar_cascade = cv.CascadeClassifier('C:/Users/Will/Downloads/Python/Computer_Vision/haar.face.xml')
features = []
labels = []

# Load training data and train classifier to establish features
def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            if img_array is None:
                continue 
                
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('Training done ---------------')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yaml')
np.save('features.npy', features)
np.save('labels.npy', labels)