import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('Mouth.xml')
cap = cv2.VideoCapture(0)
_,img=cap.read()
while 1:
    cv2.imshow('img',img)
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(img.shape)
    faces = face_cascade.detectMultiScale(gray, 1.1, 50)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x-25,y-110),(x+100,y+50),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
