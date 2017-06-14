import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import glob
import itertools

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

def loadFolder(folderPath):
    img_list = []

    for files in glob.glob(folderPath + "/*.jpg"):
        file = cv2.imread(files)
        img = cv2.resize(file, (400, 400), interpolation=cv2.INTER_AREA)
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        img_list.append(img)

    return img_list




#run through image color or grayscale histograms
def displayHist(imgList,isColor,empty,occupied,method):
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
            #img = cv2.equalizeHist(img)
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
            plt.xlim([0, 256])
            # coeff = cv2.compareHist(hist,compHist,cv2.HISTCMP_BHATTACHARYYA)
            # if((coeff-.5)<0.15):
            #     print"Empty: ",coeff
            # else:
            #     print "Occupied: ", coeff
        #if(isColor == False):
        print histClassify(img,empty,occupied,method)
        plt.show()


#for grayscale image classification
def histClassify(image,empty,occupied,method = cv2.HISTCMP_CORREL,useColor = False,bins = 256):
    if(useColor):
        emptyHist = cv2.calcHist([empty], [0,1,2], None, [bins,bins,bins], [0, bins,0,bins,0,bins])
        occupiedHist = cv2.calcHist([occupied], [0,1,2], None, [bins,bins,bins], [0, bins,0,bins,0,bins])
        imageHist = cv2.calcHist([image], [0,1,2], None, [bins,bins,bins], [0, bins,0,bins,0,bins])
    else:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        empty = cv2.cvtColor(empty, cv2.COLOR_BGR2GRAY)
        occupied = cv2.cvtColor(occupied, cv2.COLOR_BGR2GRAY)
        emptyHist = cv2.calcHist([empty], [0], None, [bins], [0, bins])
        occupiedHist = cv2.calcHist([occupied], [0], None, [bins], [0, bins])
        imageHist = cv2.calcHist([image], [0], None, [bins], [0, bins])
    coeffEmpty = cv2.compareHist(imageHist,emptyHist,method)
    coeffOccupied = cv2.compareHist(imageHist, occupiedHist, method)
    coeffSelf = cv2.compareHist(imageHist, imageHist, method)

    if(abs(coeffSelf-coeffEmpty)<abs(coeffSelf-coeffOccupied)):
        #print "empty: " +str(coeffSelf)+", " + str(abs(coeffSelf-coeffEmpty))+", "+ str(abs(coeffSelf-coeffOccupied))
        return "empty"
    else:
        #print "occupied: "+str(coeffSelf)+", "+str(abs(coeffSelf-coeffEmpty))+", "+ str(abs(coeffSelf-coeffOccupied))
        return "occupied"



#Computes pixel averages either by channel or overall - use later for logistic regression (try grayscale,color,HSV)
def computePixels(imgList):
    pixels = []
    for image in imgList:
        #pixels.append([ np.sum(image[:,:,0])/(400*400), np.sum(image[:,:,1])/(400*400), np.sum(image[:,:,2])/(400*400) ])
        pixels.append(np.sum([image[:,:,0],image[:,:,1],image[:,:,2]])/(3*400*400))
    print(pixels)
    return pixels



# images = loadFolder("dataset/empty")
# computePixels(images)
#displayHist(images,isColor = False)

#analyze using histograms
emptyLot = cv2.imread("dataset/empty/26.jpg")
emptyLot = cv2.resize(emptyLot, (400,400), interpolation=cv2.INTER_AREA)
#emptyLot = cv2.cvtColor(emptyLot,cv2.COLOR_BGR2GRAY)
occupiedLot = cv2.imread("dataset/occupied/6.jpg")
occupiedLot = cv2.resize(occupiedLot, (400,400), interpolation=cv2.INTER_AREA)
#occupiedLot = cv2.cvtColor(occupiedLot,cv2.COLOR_BGR2GRAY)

#emptyLot = cv2.equalizeHist(emptyLot)
#emptyHist = cv2.calcHist([emptyLot], [0], None, [256], [0, 256])

OPENCV_METHODS = [
	("Correlation", cv2.HISTCMP_CORREL),
	("Chi-Squared", cv2.HISTCMP_CHISQR),
	("Intersection", cv2.HISTCMP_INTERSECT),
	("Hellinger", cv2.HISTCMP_HELLINGER),
    ("Chi-Squared Alt",cv2.HISTCMP_CHISQR_ALT) ]

emptySet = loadFolder("dataset/empty")
occupiedSet = loadFolder("dataset/occupied")
#emptySet = emptySet[60:130]
#occupiedSet = occupiedSet[60:130]
#displayHist(emptySet,True,emptyLot,occupiedLot,cv2.HISTCMP_BHATTACHARYYA)

binVals = [16,32,64,128]
color = [False]
combs = list(itertools.product(binVals,color,OPENCV_METHODS))

methodAccuracies = {}
for name,method in OPENCV_METHODS:
    numCorrectE =0
    for image in emptySet:
        if(histClassify(image,emptyLot,occupiedLot,method,False,256)=="empty"):
            numCorrectE = numCorrectE+1
    print name + ": Empty Space Accuracy = "+ str(numCorrectE*1.0/len(emptySet))

    numCorrectO = 0
    for image in occupiedSet:
        if(histClassify(image,emptyLot,occupiedLot,method,False,256)=="occupied"):
            numCorrectO = numCorrectO+1
    print name + ": Occupied Space Accuracy = "+ str(numCorrectO*1.0/len(occupiedSet))

    print "Overall Accuracy: " + str((numCorrectE*1.0/len(emptySet)+numCorrectO*1.0/len(occupiedSet))/2.0)
    methodAccuracies[name] = [numCorrectE*1.0/len(emptySet),numCorrectO*1.0/len(occupiedSet)]




