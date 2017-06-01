import numpy as np
import cv2
from matplotlib import pyplot as plt


''' extract spot mannuly'''
def getSpotsCoordiantesFromImage(img, num_space) :
    coordinate_lists = []
    for i in range(num_space):
        plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        print("Please click 3 points for parking lot", i)
        coordinate = plt.ginput(3)
        print("clicked points coordinate are ", coordinate)
        coordinate_lists.append(coordinate)
    saveSpotsCoordinates(coordinate_lists)
    return coordinate_lists

''' get rotate rectangle '''
def getRotateRect(img, cooridnate_lists):
    RotateRectList = []
    

''' save the coordinate lists into txt file'''
def saveSpotsCoordinates(coordinate_lists):
    file = open("coordinates.txt","w")
    for coordinate in coordinate_lists:
        for t in coordinate:
            file.write(' '.join(str(s) for s in t) + '\n')
    file.close()

''' read txt file and load it as coordinate_lists'''
def readSpotsCoordinates(filename):
    coordinate_lists = []
    with open(filename) as file:
        count = 0
        temp_list = []
        for line in file:
            count += 1
            string_format = line.strip('\n').split(' ')
            float_format = [float(x) for x in string_format]
            temp_list.append(float_format)
            if count == 3:
                count = 0
                coordinate_lists.append(temp_list)
                temp_list = []
    return coordinate_lists

img1 = cv2.imread('parkingLot2.jpg.png', cv2.IMREAD_COLOR)
getSpotsCoordiantesFromImage(img1,2)
#coordinates_lists = readSpotsCoordinates("coordinates.txt")
print coordinates_lists
