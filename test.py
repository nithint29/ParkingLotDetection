import numpy as np
from matplotlib import pyplot as plt
import cv2


DELAY_BLUR = 500;

# Load an color image in grayscale
img = cv2.imread('parkingLot2.jpg')
img2 = cv2.imread('parkingLot2.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.subplot(151),plt.imshow(img,cmap = 'gray')
plt.title('Gray'), plt.xticks([]), plt.yticks([])

print(gray[10,10])




# apply blur and threshold
gray = cv2.GaussianBlur(gray,(15,15),0)
cv2.imshow('image1',gray)
cv2.waitKey(0)
plt.subplot(152),plt.imshow(img,cmap = 'gray')
plt.title('Blurred'), plt.xticks([]), plt.yticks([])

#thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow('image1',thresh)
cv2.waitKey(0)
plt.subplot(153),plt.imshow(img,cmap = 'gray')
plt.title('Threshold'), plt.xticks([]), plt.yticks([])
print(ret)

## resize
# height = 700.0
# r = height/ thresh.shape[1]
# dim = (int(height), int(thresh.shape[0] * r))
# thresh = cv2.resize(thresh, dim, interpolation = cv2.INTER_AREA)
# img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
# cv2.imshow("resized", thresh)
# cv2.waitKey(0)

# find contours
_, contours, _= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

validCont = []
biggest = None
max_area = 0
for i in contours:
        area = cv2.contourArea(i)
        if area > 220 and area < 10000:
                peri = cv2.arcLength(i,True)
                approx = cv2.approxPolyDP(i,0.02*peri,True)
                validCont.append(approx)
                if area > max_area and len(approx)==4:
                        biggest = approx
                        max_area = area

print (max_area)
cv2.drawContours(img, validCont, -1, (0,255,0), 3)
plt.subplot(154),plt.imshow(img,cmap = 'gray')
plt.title('Contours'), plt.xticks([]), plt.yticks([])


#cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
cv2.imshow('image1',img)
cv2.waitKey(0)

#edge detection from original
edges = cv2.Canny(img2,100,200)
cv2.imshow('image1',edges)
cv2.waitKey(0)


lines = cv2.HoughLines(edges,1,np.pi/180,200)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img2,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imshow('image1',img2)
cv2.waitKey(0)

#plt.show()
cv2.destroyAllWindows()
