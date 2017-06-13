import numpy as np
from matplotlib import pyplot as plt
import cv2


# img = cv2.imread("experiment.tif",0)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# cl1 = clahe.apply(img)
# equ = cv2.equalizeHist(img)
#
# res = np.hstack((cl1,equ)) #stacking images side-by-side
# cv2.imshow('res.png',res)
# cv2.waitKey(0)
# cv2.imshow('cl1',cl1)
# cv2.waitKey(0)
# cv2.imshow('equ',equ)
# cv2.waitKey(0)

car = cv2.imread('spots_folder/spot_'+'4.jpg',0)
space = cv2.imread('spots_folder/spot_'+'8.jpg',0)
eq = cv2.equalizeHist(car)

hist_car = cv2.calcHist([car],[0],None,[256],[0,256])
hist_space = cv2.calcHist([space],[0],None,[256],[0,256])
hist_eq = cv2.calcHist([eq],[0],None,[256],[0,256])

plt.subplot(221), plt.imshow(car, 'gray')
plt.subplot(222), plt.plot(hist_car)

plt.subplot(223), plt.imshow(eq, 'gray')
plt.subplot(224), plt.plot(hist_eq)


plt.show()


# Numpy equalization
