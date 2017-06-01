import numpy as np
import cv2


class PolygonDrawer(object):
    polyPoints = [None]*4
    i = 0
    
    def __init__(self,windowName,image):
        self.originalImage = image
        self.image = np.copy(self.originalImage)
        #self.image = image
        self.windowName = windowName
        self.POINTS = []
        self.i = 0



    #Called on mouse click
    #left click to place point, right click to add polygon
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
            self.POINTS.append(self.polyPoints)
            self.polyPoints = [None]*4
            self.i = 0
            print(self.POINTS)

        #delete current points
        elif event == cv2.EVENT_RBUTTONDBLCLK:
            print "deleting"
            self.polyPoints = [None]*4
            self.i = 0;


    def run(self):
        cv2.namedWindow(self.windowName)
        cv2.setMouseCallback(self.windowName,self.place_poly)

        while(1):

            if(self.i>=1):
                pts = np.array(self.polyPoints[0:self.i],np.int32)#.reshape((-1,1,2))
                cv2.polylines(self.image,[pts],False,(255,255,255))
                #cv2.line(self.image,self.polyPoints[-1],self.current,(255,255,255))
            
            cv2.imshow(self.windowName,self.image)

            key = cv2.waitKey(20)

            if key & 0xFF == 27:
                self.saveSpotsCoordinates()
                break
            elif key & 0xFF == 32:
                self.loadPointsOnImage()

        cv2.destroyAllWindows()
        return

    def loadPointsOnImage(self):
        for polygon in self.POINTS:
            if(polygon !=None):
                for point in polygon:
                    self.image = np.copy(self.originalImage)
                    cv2.circle(self.image, (point[0],point[1]), 5, (255, 0, 0), -1)

    def loadFromText(self):
        print "hi"


    def saveSpotsCoordinates(self):
        file = open("coordinates.txt", "w")
        for polygon in self.POINTS:
            for t in polygon:
                file.write(' '.join(str(s) for s in t) + '\n')
        file.close()



if __name__ == "__main__":
    print("hello world")
    img = cv2.imread("parkingLot.jpg")
    p = PolygonDrawer("poly",img)
    p.run()
    cv2.imshow("image",img)
    cv2.waitKey(0)
