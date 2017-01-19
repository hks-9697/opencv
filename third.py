import cv2
import numpy as np


vid=cv2.VideoCapture(0)
template = cv2.imread('web1.jpg',0)
w, h = template.shape[::-1]



while True:
        _,frame=vid.read()
        img_rgb = frame
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.4
        loc = np.where( res >= threshold)

        for pt in zip(*loc[::-1]):
                cv2.rectangle(img_rgb, pt, (pt[0] + 100, pt[1] + 100), (255,255,255), 0.2)

        cv2.imshow('Detected',img_rgb)
        
        
        k=cv2.waitKey(5) & 0xFF
        if k==27 :
                break
cv2.destroyAllWindows()
vid.release()





