
from itertools import count
import cv2, os
from datetime import datetime as dt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "dataset")


video = cv2.VideoCapture(0)

video.set(3, 640)
video.set(4, 480) 

face_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")


label_name = input("Enter your name: ")

get_date = str(dt.now()).split(" ")[0]
get_time = str(dt.now()).split(" ")[1].replace(":", "-")[:8]

label_dir = label_name + "." + get_date + "-" + get_time
try:
    if os.path.dirname(__file__) == BASE_DIR and label_name != None:
        os.chdir(image_dir)
        os.mkdir(label_dir)
except FileNotFoundError:
    print("Invalid name")
    

os.chdir(BASE_DIR)
print(os.getcwd())

count = 1
while True:
    check, frame = video.read()

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray_img,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )

    color = (255,0,0) # BGR, 0-255
    line_thickness = 2


    for (x,y,w,h) in faces:
        roi = gray_img[y:y+h, x:x+w] # region of interest
        cv2.rectangle(frame, (x,y), (x+w, y+h), color, line_thickness)
        cv2.imwrite(f"dataset/{label_dir}/{label_name}.{count}.jpg", roi)
        count += 1
    cv2.imshow("Video", frame)
    key = cv2.waitKey(100)

    if key == ord("q") or key == ord("Q"):
        break
    elif count >= 30:
        break

video.release()
cv2.destroyAllWindows