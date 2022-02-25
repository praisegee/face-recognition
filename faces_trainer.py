import os, cv2, pickle
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "dataset")

# cascade for faces detection
face_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")
# create the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()


current_id = 0
label_ids = {}
train = []
labels = []
for root, dirs, files in os.walk(image_dir):
    for file in files:
        path = os.path.join(root, file)
        label = os.path.basename(root).split(".")[0].title()
        # print(path, label)
        if not label in label_ids:
            label_ids[label] = current_id
            current_id += 1
        id = label_ids[label]
        PIL_image = Image.open(path).convert("L") # to grayscale
        np_image = np.array(PIL_image, "uint8") # to numpy array
        # print(np_image)

        faces = face_cascade.detectMultiScale(np_image, scaleFactor=1.5, minNeighbors=5)

        for (x,y,w,h) in faces:
            roi = np_image[y:y+h, x:x+w] # region of interest
            train.append(roi)
            labels.append(id)


with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)


print("[INFO:] Training in progress...")
recognizer.train(train, np.array(labels))
recognizer.save("trainner/trainner.yml")





