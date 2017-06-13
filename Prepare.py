import numpy as np
from matplotlib import pyplot as plt
import cv2
import os

def readFromDict(imageDict):
    for i in imageDict:
        cv2.imshow("lot"+str(i),imageDict[i][1])
        cv2.waitKey(0)
    return

#Generates list of images from folder (resized to 400 X 400)
def readFromFolder(folder):
    img_list = []
    for image in os.listdir(folder):

        img = cv2.imread(os.path.join(folder,image))
        img = cv2.resize(img, (400,400), interpolation=cv2.INTER_AREA)
        if image is not None:
            img_list.append(img)
            # print( np.sum([img[:,:,0],img[:,:,1],img[:,:,2]])/(3*400*400) )
    return img_list

#run through image color histograms
def displayHist(imgList,isColor):
    for img in imgList:
        # img[:,:,0] = cv2.equalizeHist(img[:,:,0])
        # img[:, :, 1] = cv2.equalizeHist(img[:,:,1])
        # img[:, :, 2] = cv2.equalizeHist(img[:,:,2])
        if(isColor == False):
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        plt.subplot(221), plt.imshow(img, 'gray')
        plt.subplot(222),

        if(isColor):
            colors = ('b', 'g', 'r')
        else:
            colors = ('g')
        for i, col in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
            plt.xlim([0, 256])

        plt.show()






def computePixels(imgList):
    pixels = []
    for image in imgList:
        #pixels.append([ np.sum(image[:,:,0])/(400*400), np.sum(image[:,:,1])/(400*400), np.sum(image[:,:,2])/(400*400) ])
        pixels.append(np.sum([image[:,:,0],image[:,:,1],image[:,:,2]])/(3*400*400))
    print(pixels)
    return pixels

images = readFromFolder("spots_folder")
computePixels(images)
displayHist(images,isColor = True)

