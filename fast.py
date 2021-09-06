import numpy as np
import cv2
import face_recognition
import pickle5 as pickle

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
face_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)

labels = {}
with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cas.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)

    for(x, y, w, h) in faces:
        print(x, y, w, h)

        roi_gray = gray[y:y+h, x:x+w]
        roi_colour = frame[y:y+h, x:x+w]

        id, conf = recognizer.predict(roi_gray)

        print(conf)
        font = cv2.FONT_HERSHEY_SIMPLEX
        colour = (255,0,255)
        stroke = 2
        name = labels[id]
        if conf > 50 :
            cv2.putText(frame, name, (x,y), font, 1, colour, stroke)
        
        #img_item = "my-"

        #finding the faces
        
        end_coordinate_x = x + w
        end_cooridate_y = y + h
        colour = (225,0,225)

        #displaying box around face
        cv2.rectangle(frame, (x, y), (end_coordinate_x, end_cooridate_y), colour, stroke)
        #cv2.rectangle(frame, (x-35, y-35), (end_coordinate_x, end_cooridate_y), colour, cv2.FILLED) find naming techique    
        
    cv2.imshow('frame', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()