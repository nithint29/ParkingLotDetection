import numpy as np
import cv2

from PolygonDrawer import *
from parkingSpaceExtraction import *
from detection import *
from Prepare import *

class SmartParking:
    '''
    create smart parking class
    it can be initalized from image list with file folder
    or it can be initialzed from file folder
    '''
    def __init__(self):
        self.current_image = None
        self.current_pos = 0
        self.__file_folder = ""
        self.__camera_position = []
        self.__spots_list = []
        self.__current_coordinate_list = None
        self.__svm_model = None
        self.__lr_model = None
        self.__usesvm = True
        self.__uselr = False

    #initialize from image list and file folder
    def initial(self, img_list, file_folder):
        #assign the svm model classifier
        self.__svm_model = SVM(file_path="svm_model1.xml")
        #initialize from image list
        self.__file_folder = file_folder
        for i, img in enumerate(img_list):
            coordinate_path = file_folder + "/coordinates/coordinate_" + "{:03d}".format(i) + ".txt"
            p = PolygonDrawer("poly", img,coordinate_path,file_folder+"/spots_folder")
            p.run()
            self.__camera_position.append(i)



    #process with the input image and positions
    def process(self, img, pos):
        self.current_image = img
        self.current_pos = pos
        # find the correspond camera position's ROI coordinates
        file_name = self.__file_folder + "/coordinates/coordinate_" + "{:03d}".format(pos) + ".txt"
        self.__current_coordinate_list = readSpotsCoordinates(file_name)
        print self.__current_coordinate_list
        self.__spots_list = getRotateRect(self.current_image, self.__current_coordinate_list)

        length = len(self.__spots_list)
        print(length)

        emptySpots = []
        # detect if the spot is empty or not
        for i, spot in enumerate(self.__spots_list):
            if(self.__usesvm):
                spot_feature = getHistFeature(spot, bins=16, usecolor=True)
                spot_feature = spot_feature.reshape((1,len(spot_feature)))
                isEmpty = self.__svm_model.predict(spot_feature)
                if(isEmpty[0] == 0):
                    plt.imshow(spot, cmap = 'gray', interpolation = 'bicubic')
                    plt.xticks([]), plt.yticks([])
                    plt.show()
                    emptySpots.append(i)
                    print ("Spot " + str(self.current_pos) + "-" + str(i) + " is empty")
            elif(self.__uselr):
                spot_feature = getHistFeature(spot, bins=16, usecolor=True)
                isEmpty = self.__lr_model.predict(spot)
                if(isEmpty):
                    emptySpots.append(i)
                    print ("Spot " + str(self.current_pos) + "-" + str(i) + " is empty")
        return emptySpots




if __name__ == "__main__":
    s = SmartParking()
    img_list = []
    img0 = cv2.imread("parkingImage/view7.png", cv2.IMREAD_COLOR)
    img_list.append(img0)
    img_list.append(cv2.imread("parking_example.png",cv2.IMREAD_COLOR))
    s.initial(img_list, "test")
    # s.initialFromFolder("test")
    s.process(img0,0)