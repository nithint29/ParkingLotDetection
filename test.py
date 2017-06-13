import numpy as np
from matplotlib import pyplot as plt
import cv2
import collections

POINTS = [];
rectPoints  = [];
DELAY_BLUR = 500;
global img

def placeRects(image):
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', place_rect)

    while (1):
        cv2.imshow('image', image)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    return

#called when mouse is pressed
def place_rect(event,x,y,flags,param):
    if event == cv2.EVENT_FLAG_LBUTTON:
        cv2.circle(img,(x,y),3,(255,0,0),-1)
        rectPoints.append([x,y])
        print(rectPoints)
    elif event == cv2.EVENT_FLAG_RBUTTON:
        pts = np.array(rectPoints,np.int32).reshape((-1,1,2))
        cv2.polylines(img,[pts],True,(255,255,255))
        print(POINTS)


img = cv2.imread('CARS.jpg')
img2 = cv2.imread('CARS.jpg')

#place polygons on image img
placeRects(img)



# Load a color image in grayscale
img = cv2.imread('output.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.subplot(151),plt.imshow(img,cmap = 'gray')
plt.title('Gray'), plt.xticks([]), plt.yticks([])


# apply blur and threshold
gray = cv2.GaussianBlur(gray,(15,15),0)
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

#create a mask from contours
mask = np.zeros(img.shape, np.uint8)
cv2.drawContours(mask, validCont, -1, (255,255,255),-1)
cv2.imshow('Mask',mask)

cv2.drawContours(img, contours, -1, (0,255,0), 3)
plt.subplot(154),plt.imshow(img,cmap = 'gray')
plt.title('Contours'), plt.xticks([]), plt.yticks([])


#show contours
#cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
cv2.imshow('Contours',img)
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
