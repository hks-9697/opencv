import cv2
import numpy as num
import matplotlib as math
vid=cv2.VideoCapture(0)
_,initial=vid.read()
#fgbg = cv2.BackgroundSubtractorMOG()
#C=initial.copy()

face_cascade = cv2.CascadeClassifier('parojos.xml')

initial=cv2.cvtColor(initial,cv2.COLOR_BGR2GRAY)
while True:
        _,frame=vid.read()
        #hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        #lowerbrown=num.array([0,50,50])
        #upperbrown=num.array([25,190,255])
        black=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #mask=cv2.inRange(hsv,lowerbrown,upperbrown)
        kernal=num.ones((25,25),num.uint8)
        kernal4=num.ones((10,10),num.uint8)
        kernal2=num.ones((2,25),num.uint8)
        kernal3=num.ones((25,2),num.uint8)
        mask2=cv2.absdiff(initial,black)
        #initial=cv2.addWeighted(initial,0.96,black,0.04,0)
        mask2=cv2.erode(mask2,kernal4,iterations= 1)
        cv2.imshow('diff',mask2)
        mask3=cv2.GaussianBlur(mask2,(15,15),0)
        mask3=cv2.GaussianBlur(mask2,(15,15),0)
        cv2.imshow('newmask',mask3)
        #frameDelta=fgbg.apply(mask3)
        frameDelta=mask3     
        	# compute the absolute difference between the current frame and
	# first frame
	thresh = cv2.threshold(frameDelta, 18, 255, cv2.THRESH_BINARY)[1]
	#thresh = cv2.adaptiveThreshold(frameDelta, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
	thresh=cv2.dilate(thresh,kernal,iterations=2)
	cv2.imshow('Adaptive threshold',thresh)
 
	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	#thresh = cv2.erode(thresh, None, iterations=2)
	(cnts, _) = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	counter=0
 
	try:
	# loop over the contours
		for c in cnts:
	        	#counter=counter+1
			# if the contour is too small, ignore it
			if (cv2.contourArea(c) < 20000):
				#counter=counter-1
				continue
				
			# compute the bounding box for the contour, draw it on the frame,
			# and update the text
			(x, y, w, h) = cv2.boundingRect(c)
			#print(x)
			person=frame[y:y+h,x:x+w]
			k=cv2.rectangle(frame, (x, y), (x + w, y + h), (244, 255, 0), 1)
			#print(k)
			cv2.imshow("person",person)
			gray = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
    			faces = face_cascade.detectMultiScale(gray, 1.05, 10)
    			for (x1,y1,w1,h1) in faces:
        			cv2.rectangle(person,(x1-25,y1-110),(x1+w1+50,y1+h1+25),(255,0,0),2)
        			counter=counter+1
			#cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		counter=counter+ord('0')
		font = cv2.FONT_HERSHEY_SIMPLEX
        	cv2.putText(frame,'no of people '+chr(counter),(0,130), font, 1, (200,255,155), 2)
	except (Exception,RuntimeError, TypeError, NameError):
		pass
	
		
#show frame
        cv2.imshow('original',frame)
 
        
        k=cv2.waitKey(5) & 0xFF
        if k==27 :
                break
cv2.destroyAllWindows()
vid.release()
              
