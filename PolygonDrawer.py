import numpy as np
import cv2


class PolygonDrawer(object):
    polyPoints = [0]*4
    i = 0
    
    def __init__(self,windowName,image):
        self.image = image
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
                cv2.circle(img,(x,y),5,(255,0,0),-1)
                self.polyPoints[self.i] = (x,y)
                self.i = self.i+1
                print(self.polyPoints)
                
        elif event == cv2.EVENT_FLAG_RBUTTON and self.i == 4:
            pts = np.array(self.polyPoints,np.int32)#.reshape((-1,1,2))
            cv2.polylines(self.image,[pts],True,(255,255,255))
            self.POINTS.append(self.polyPoints)
            self.polyPoints = [0]*4
            self.i = 0
            print(self.POINTS)

    def run(self):
        cv2.namedWindow(self.windowName)
        cv2.setMouseCallback(self.windowName,self.place_poly)

        while(1):
            #pts = np.array(self.polyPoints,np.int32)#.reshape((-1,1,2))
            #cv2.polylines(self.image,[pts],True,(255,255,255))
            
            cv2.imshow(self.windowName,self.image)
            if cv2.waitKey(20) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
        return



if __name__ == "__main__":
    print("hello world")
    img = cv2.imread("parkingLot.jpg")
    p = PolygonDrawer("poly",img)
    p.run()
