import numpy as np
import cv2
DELAY_BLUR = 500;

# Load an color image in grayscale
img = cv2.imread('parkingLot.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)





gray = cv2.GaussianBlur(gray,(17,17),0)
thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

cv2.imshow('image1',thresh)
cv2.waitKey(0)

_, contours, _= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

biggest = None
max_area = 0
for i in contours:
        area = cv2.contourArea(i)
        if area > 100:
                peri = cv2.arcLength(i,True)
                approx = cv2.approxPolyDP(i,0.02*peri,True)
                if area > max_area and len(approx)==4:
                        biggest = approx
                        max_area = area


cv2.drawContours(img, contours, 120, (0,255,0), 3)



height = 700.0
r = height/ img.shape[1]
dim = (int(height), int(img.shape[0] * r))
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
#cv2.imshow("resized", resized)
#cv2.waitKey(0)

#cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
cv2.imshow('image1',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.__version__
