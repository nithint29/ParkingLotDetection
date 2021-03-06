import numpy as np
import cv2
import os
import errno

class PolygonDrawer(object):
    """Takes an image and allows user to draw and extract polygons from it. Polygon data is stored in input text fileName.
        Images extracted are saved in input folderName."""
    polyPoints = [None]*4
    i = 0
    FILE_NAME = "coordinates.txt"
    PICTURE_FOLDER = "spots_folder"
    
    def __init__(self,windowName,image,fileName = FILE_NAME,folderName = PICTURE_FOLDER):
        """Initialize object on a given image with filName to store coordinates and folderName to store each space image"""
        self.originalImage = image
        self.image = np.copy(self.originalImage)
        #self.image = image
        self.windowName = windowName
        self.POINTS = []
        self.i = 0
        self.spaceMap = {}
        self.FILE_NAME = fileName
        self.PICTURE_FOLDER = folderName
        #self.ensure_dir(self.PICTURE_FOLDER)
        #self.ensure_dir(self.FILE_NAME)

    def reInit(self,image,fileName = FILE_NAME,folderName = PICTURE_FOLDER):
        """Use to reselect the image to process, with new output coordinate file and picture folder"""
        self.originalImage = image
        self.image = np.copy(self.originalImage)
        self.FILE_NAME = fileName
        self.PICTURE_FOLDER = folderName
        self.POINTS = []
        self.i = 0


    #Called on mouse click
    #left click to place point, right click to add polygon, double right click to delete current selection
    def place_poly(self,event,x,y,flags,param):

        if event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)
        
        if event == cv2.EVENT_FLAG_LBUTTON:
            if self.i<4:
                cv2.circle(self.image,(x,y),5,(255,0,0),-1)
                self.polyPoints[self.i] = (x,y)
                self.i = self.i+1
                print(self.polyPoints)
                
        elif event == cv2.EVENT_FLAG_RBUTTON and self.i == 4:
            pts = np.array(self.polyPoints,np.int32)#.reshape((-1,1,2))
            #cv2.polylines(self.image,[pts],True,(255,255,255))
            cv2.fillPoly(self.image,[pts],(255,255,255))
            font = cv2.FONT_HERSHEY_SIMPLEX
            avg_point = np.mean(pts,axis=0)
            print(avg_point)
            cv2.putText(self.image, str(len(self.POINTS)), tuple(avg_point.astype(int)), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            self.POINTS.append(self.polyPoints)
            self.polyPoints = [None]*4
            self.i = 0
            #print(self.POINTS)

        #delete current points
        elif event == cv2.EVENT_RBUTTONDBLCLK:
            print "deleting"
            self.polyPoints = [None]*4
            self.i = 0;


    def run(self,saveImages=True):
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.windowName,self.place_poly)

        while(1):
            #to draw polygon in progress
            if(self.i>=1):
                pts = np.array(self.polyPoints[0:self.i],np.int32)#.reshape((-1,1,2))
                cv2.polylines(self.image,[pts],False,(255,255,255))
                #cv2.line(self.image,self.polyPoints[-1],self.current,(255,255,255))
            
            cv2.imshow(self.windowName,self.image)

            key = cv2.waitKey(20)
            if key & 0xFF == 27:
                self.saveSpotsCoordinates()
                #cv2.imwrite("labeledPic.jpg", self.image)
                if(saveImages==True):
                    self.saveImageList(self.getRotateRect(self.POINTS))
                break

            #press spacebar to read spot info from file onto image
            elif key & 0xFF == 32:
                self.POINTS=[]
                self.readSpotsCoordinates(self.FILE_NAME)
                self.loadPointsOntoImage()
                #cv2.imwrite("labeledPic.jpg",self.image)

        cv2.destroyAllWindows()
        return


    #
    def loadPointsOntoImage(self):
        self.image = np.copy(self.originalImage)
        for i,polygon in enumerate(self.POINTS):
            pts = np.array(polygon, np.int32)
            cv2.fillPoly(self.image, [pts], (255, 255, 255))
            font = cv2.FONT_HERSHEY_SIMPLEX
            avg_point = np.mean(pts, axis=0)
            cv2.putText(self.image, str(i), tuple(avg_point.astype(int)), font, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)
            if(polygon !=None):
                for point in polygon:
                    cv2.circle(self.image, (int(point[0]),int(point[1])), 5, (255, 0, 0), -1)

    def four_point_transform(self, coordinate):
        (tl, tr, br, bl) = coordinate

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # create new image has same size as we calculate before
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # change original points list to nparray and set to the same dtype
        rect = np.asarray(coordinate).astype(dtype="float32")
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(self.originalImage, M, (maxWidth, maxHeight))

        # return the warped image
        return warped

    #creates list of warped images from POINTS
    def getRotateRect(self,pointList):
        warped_img_lists = []
        for coordinate in pointList:
            warped = self.four_point_transform(coordinate)
            # plt.imshow(warped, cmap='gray', interpolation='bicubic')
            # plt.xticks([]), plt.yticks([])
            # plt.show()
            warped_img_lists.append(warped)
        return warped_img_lists


    #save polygon data to text file
    def saveSpotsCoordinates(self):
        file = open(self.FILE_NAME, "w")
        for polygon in self.POINTS:
            for t in polygon:
                file.write(' '.join(str(s) for s in t) + '\n')
        file.close()
        print("Coordinates saved successfully")

    #load polygon data points from text file into POINTS
    def readSpotsCoordinates(self,filename):
        self.POINTS = []
        with open(filename) as file:
            count = 0
            temp_list = []
            for line in file:
                count += 1
                string_format = line.strip('\n').split(' ')
                float_format = [float(x) for x in string_format]
                temp_list.append(float_format)
                # every 4 points as a rectangle
                if count == 4:
                    count = 0
                    self.POINTS.append(temp_list)
                    temp_list = []
        #print("read coordinates lists successfully:")
        #print(self.POINTS)
        return self.POINTS


    def saveImageList(self,img_list):
        """Saves images in list to the PICTURE_FOLDER"""
        for i, img in enumerate(img_list):
            cv2.imwrite(self.PICTURE_FOLDER + "/spot_" + str(i) + ".jpg", img)
        print("saved N = " + str(len(img_list)) + " images in path " + self.PICTURE_FOLDER)

    def ensure_dir(self,filename):
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

    def setCoordinatesFile(self,fileName = None,folderName = None):
        #self.ensure_dir(fileName)
        #self.ensure_dir(folderName)
        if(fileName != None):
            self.FILE_NAME = fileName
        if(folderName != None):
            self.PICTURE_FOLDER = folderName

if __name__ == "__main__":
    #print("hello world")
    img = cv2.imread("camera002.jpg")
    p = PolygonDrawer("poly",img,"coordinates.txt","spots_folder")
    p.run(saveImages=False)


