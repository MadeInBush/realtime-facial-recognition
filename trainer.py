import os
import numpy as np
import cv2
from PIL import Image
from numpy.lib.type_check import imag
import pickle5 as pickle

directory = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(directory, "images")

face_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()




filename_id = {}
filenames = []
x_train = []
current_id = 0


for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file) # finding file path
            filename = os.path.basename(root).replace(" ", "_").lower()
            
            if not filename in filename_id:
                filename_id[filename] = current_id
                current_id += 1

            id_ = filename_id[filename]
            pil_image = Image.open(path).convert("L") # saving the path as a grayscale image
            
            image_array = np.array(pil_image, "uint8")

            faces = face_cas.detectMultiScale(image_array, scaleFactor = 1.5, minNeighbors = 5)

            for(x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                filenames.append(id_)

with open("labels.pickle", "wb") as f:
    pickle.dump(filename_id, f)

print(current_id)

recognizer.train(x_train, np.array(filenames))
recognizer.save("trainner.yml")

#print(recognizer)
#print(filename_id)
                        