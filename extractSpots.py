import cv2
import numpy as np
import math
import sys

print(sys.version)

def slope((x1,y1),(x2,y2)):
    return float(y2-y1)/float(x2+.001-x1)
def myThresh(img):

    return
def myCorners(img):
    return


img = cv2.imread("parkingLot.jpg")
#img = cv2.resize(img,(400,400))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(15,15),0)
ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
thresh2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,2)

#erode
kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(thresh,kernel,iterations=1)

corners = cv2.goodFeaturesToTrack(erosion,60,0.01,10)

for i in corners:
    x,y = i.ravel()
    cv2.circle(gray,(x,y),3,255,-1)

#cv2.imshow("Thresholded",gray)
#cv2.waitKey(0)


img = cv2.imread("parkingExample.jpg")
#img = cv2.imread("parkingLot.jpg")
#img = cv2.resize(img,(400,400))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#gray = cv2.GaussianBlur(gray,(5,5),0)
gray = cv2.bilateralFilter(gray,5,25,25)
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
thresh2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,2)

edges = cv2.Canny(gray,100,200)
#erode
kernel = np.ones((3,3),np.uint8)
dilate = cv2.dilate(edges,kernel,iterations=1)
erosion = cv2.erode(dilate,kernel,iterations=1)
#erosion = cv2.medianBlur(erosion,5)

corners = cv2.goodFeaturesToTrack(gray,200,0.05,10)

for i in corners:
    x,y = i.ravel()
    cv2.circle(gray,(x,y),4,255,-1)
    closeX = -1
    closeY = -1
    minDist = 1000000
    for j in corners:
        x2,y2 = j.ravel()
        dist =math.sqrt((x-x2)**2 + (y-y2)**2)
        if(dist>0 and dist<minDist):
            minDist = dist
            closeX = x2
            closeY = y2
    sl = slope((x,y),(closeX,closeY))
    lineX = 1000
    lineY = y+sl*(1000-x)
    cv2.line(gray,(x,y),(int(closeX),int(closeY)),(255,255,255),thickness=2)
    cv2.line(gray,(x,y),(int(x+(x-closeX)),int(y+(y-closeY))),(255,255,255),thickness=2)


ret,thresh = cv2.threshold(gray,80,255,cv2.THRESH_BINARY_INV)
thresh2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,2)

#find and process contours
_, contours, _= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))

validCont = []
biggest = None
max_area = 0
for i in contours:
        area = cv2.contourArea(i)
        if area > 220 and area < 3000:
                peri = cv2.arcLength(i,True)
                approx = cv2.approxPolyDP(i,0.02*peri,True)
                validCont.append(approx)
                if area > max_area and len(approx)==4:
                        biggest = approx
                        max_area = area

print (max_area)

for i in validCont:
    rect = cv2.minAreaRect(i)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

#cv2.drawContours(img, validCont, -1, (0,255,0),2)
cv2.imshow("Thresholded",gray)
cv2.waitKey(0)


#harris corner detection
img = cv2.imread("parkingExample.jpg")
#img = cv2.resize(img,(400,400))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(15,15),0)
ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(thresh,kernel,iterations=1)

gray = np.float32(erosion)
dst = cv2.cornerHarris(gray,3,3,0.04)
#result is dilated for marking the corners, not important
kernel = np.ones((5,5),np.uint8)
dst = cv2.dilate(dst,kernel,iterations=1)
# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.4*dst.max()]=[255,255,255]

cv2.imshow('Harris',img)
cv2.waitKey(0)
