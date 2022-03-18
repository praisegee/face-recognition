import os, pickle
import cv2


# cascade for faces detection
face_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")
# create the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner/trainner.yml")


labels = {}
with open("labels.pickle", "rb") as f:
    labels = pickle.load(f)
    labels = {v:k for k,v in labels.items()} # inverting the keys and values for the labels


video = cv2.VideoCapture(0) # capture images from the webcam


while True:
    check, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # flip = cv2.flip(frame, 2) # mirror the image L-R

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)


    for (x,y,w,h) in faces:
        roi = gray[y:y+h, x:x+w] # region of interest

        id, conf = recognizer.predict(roi)
        # print(id, conf)
        # print(labels[id])

        if conf < 100:
            name = labels[id]
            conf = " {0}%".format(round(100-conf))
        else:
            name = "unknow face"
            conf = " {0}%".format(round(100-conf))

        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255,255,255)
        cv2.putText(frame, name, (x+5,y-5), font, 1, color, 2)
        cv2.putText(frame, str(conf), (x+5,y+h-5), font, 1,   color, 1)
        


        color = (255,0,0) # BGR, 0-255
        line_thickness = 2

        cv2.rectangle(frame, (x,y), (x+w, y+h), color, line_thickness)
        

    cv2.imshow("Frames", frame)
    key = cv2.waitKey(1)
    if key == ord("q") or key == ord("Q"):
        break
    

video.release()
cv2.destroyAllWindows()