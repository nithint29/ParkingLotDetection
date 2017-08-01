import numpy as np
import cv2

from PolygonDrawer import *
from parkingSpaceExtraction import *
from detection import *
from Prepare import *
import os
import pickle
from time import sleep
from onvif import ONVIFCamera

class SmartParking:
    '''
    create smart parking class
    it can be initalized from image list with file folder
    or it can be initialzed from file folder
    '''
    def __init__(self):
        self.current_image = None
        self.current_pos = 0
        self.__file_folder = "test"
        self.__camera_position = []
        self.__spots_list = []
        self.__current_coordinate_list = None
        self.__svm_model = None
        self.__lr_model = None
        self.__usesvm = False
        self.__uselr = True
        pkl = open('LR.pkl','rb')
        self.thetaFinal = pickle.load(pkl)
        pkl.close()
        #initialize camera setting
        self.streamURL = "rtsp://bigbrother.winlab.rutgers.edu/stream1"
        self.mycam = ONVIFCamera('192.168.204.111', 80, 'admin', 'admin', 'C:/Users/basis_000/Anaconda2/wsdl/')
        self.pos = [{'_x': -0.2, '_y': 0.5} , {'_x': -0.03, '_y': 0.55}, {'_x': 0.07, '_y': 0.6}]

    #initialize from image list and file folder
    def initial(self, img_list, file_folder):
        #initialize from image list
        self.__file_folder = file_folder

        for i, img in enumerate(img_list):
            mydir =os.getcwd()+os.sep+file_folder
            coordpath = mydir+"/coordinates/"
            if(os.path.exists(coordpath)==False):
                os.makedirs(os.path.dirname(coordpath))
            coordinate_path = coordpath+"coordinate_" + "{:03d}".format(i) + ".txt"
            if(os.path.exists(coordinate_path)==False):
                file = open(coordinate_path,'w+')
                file.close()
            if (os.path.exists(mydir+"/spots_folder{}".format(i)) == False):
                os.makedirs(mydir+"/spots_folder{}".format(i))
                print("making folders")
            p = PolygonDrawer("poly", img,coordinate_path,mydir+"/spots_folder{}".format(i))
            p.run()
            # p.readSpotsCoordinates(coordinate_path)
            # p.saveImageList()
            self.__camera_position.append(i)

        #assign the svm model classifier
        if(self.__usesvm):
            self.__svm_model = SVM(file_path="svm_model1.xml")
        elif(self.__uselr):
            pkl = open('LR.pkl', 'rb')
            self.thetaFinal = pickle.load(pkl)
            pkl.close()


    #process with the input image and positions
    def process(self, img, pos):
        self.current_image = img
        self.current_pos = pos
        # find the correspond camera position's ROI coordinates
        file_name = self.__file_folder + "/coordinates/coordinate_" + "{:03d}".format(pos) + ".txt"
        self.__current_coordinate_list = readSpotsCoordinates(file_name)
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
                    # plt.imshow(spot, cmap = 'gray', interpolation = 'bicubic')
                    # plt.xticks([]), plt.yticks([])
                    # plt.show()
                    emptySpots.append(i)
                    print ("Spot " + str(self.current_pos) + "-" + str(i) + " is empty")
            elif(self.__uselr):
                status = predict(spot,self.thetaFinal,32,True,True)
                if(status == 0):
                    emptySpots.append(i)
                    print ("Spot " + str(self.current_pos) + "-" + str(i) + " is empty - LR")
                    pts = np.array(self.__current_coordinate_list[i], np.int32)  # .reshape((-1,1,2))
                    cv2.fillPoly(self.current_image, [pts], (0, 255, 0))
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    avg_point = np.mean(pts, axis=0)
                    cv2.putText(self.current_image, str(i), tuple(avg_point.astype(int)), font, 1, (0, 0, 255),
                                2, cv2.LINE_AA)
                    cv2.imwrite("./static/images/labeled{:03d}.jpg".format(pos), self.current_image)
                    # cv2.imshow("Empty Spot",spot)
                    # cv2.waitKey(0)
        return emptySpots

    #camera control
    def getImageFromCamera(self):
        # Create media service object
        media = self.mycam.create_media_service()
        # Create ptz service object
        ptz = self.mycam.create_ptz_service()
        # Get target profile
        media_profile = media.GetProfiles()[0]
        request = ptz.create_type('AbsoluteMove')
        request.ProfileToken = media_profile._token

        img_list = []
        #pos 0
        for i,pos in enumerate(self.pos):
            request.Position.PanTilt = pos
            ptz.AbsoluteMove(request)
            sleep(2)
            cap = cv2.VideoCapture(self.streamURL)
            ret, frame = cap.read()
            cv2.imwrite("./static/images/camera" + "{:03d}".format(i) + ".jpg", frame)
            img_list.append(frame)
        return img_list




if __name__ == "__main__":
    s = SmartParking()
    emptySpots = []
    parkingInfo = []
    img_list = []

    # img_list.append(img0)
    # img_list.append(img1)
    # img_list.append(img2)
    s.initial(img_list, "test")
    s.getImageFromCamera()
    img_name_list = ["./static/images/camera000.jpg", "./static/images/camera001.jpg", "./static/images/camera002.jpg"]
    #img_name_list = ["./labeled000.jpg", "./labeled001.jpg", "./labeled002.jpg"]
    img0 = cv2.imread(img_name_list[0], cv2.IMREAD_COLOR)
    img1 = cv2.imread(img_name_list[1], cv2.IMREAD_COLOR)
    img2 = cv2.imread(img_name_list[2], cv2.IMREAD_COLOR)

    img_list.append(img0)
    img_list.append(img1)
    img_list.append(img2)
    s.initial(img_list, "test")

    for i in range(3):
        temp = s.process(img_list[i], i)
        if len(temp) == 0:
            info = "There are no empty spots in parking lot " + str(i) + "."
        else:
            info = ""
            for j, spot in enumerate(temp):
                info = info + "spot" + str(i) + "_" + str(spot)
                if j != len(temp) - 1:
                    info += ", "
            info += " are empty"
        parkingInfo.append(info)
    print parkingInfo

