import numpy as np
from matplotlib import pyplot as plt
import cv2

car = cv2.imread('spots_folder/spot_'+'0.jpg',0)
space = cv2.imread('spots_folder/spot_'+'1.jpg',0)

hist_car = cv2.calcHist([car],[0],None,[256],[0,256])
hist_space = cv2.calcHist([space],[0],None,[256],[0,256])

plt.subplot(221), plt.imshow(car, 'gray')
plt.subplot(222), plt.plot(hist_car)

plt.subplot(223), plt.imshow(space, 'gray')
plt.subplot(224), plt.plot(hist_space)


plt.show()
