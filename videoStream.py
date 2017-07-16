from Prepare import *
from PolygonDrawer import *
from time import  time

#numSamples = 150
frameNum=0
timeInterval = 60 #number of minutes between each capture
auto =True
startTime = time()

#train on rawdata
thetaFinal = trainOnFolder("rawdataset/empty","rawdataset/occupied",-1,32,True,True,lam=100)
testFolder = loadFolder("spots_folder")

#start stream
cap = cv2.VideoCapture("rtsp://bigbrother.winlab.rutgers.edu/stream1")
initial = True
initState = []
ret, prevFrame = cap.read()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    p = PolygonDrawer("frame", frame, "coordinates.txt", "spots_folder")
    coordList = p.readSpotsCoordinates("coordinates.txt")

    if(initial):
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('frame', 600, 600)
        if(auto==False):
            p.run(saveImages=False)

        if(auto == True):
            cv2.imwrite("spots_folder/frame{}.jpg".format(frameNum), frame)
            frameNum += 1
            testImages = p.getRotateRect(coordList)

            for i, img in enumerate(testImages):
                isOcc = predict(img, thetaFinal, 32, True, True)
                initState.append(isOcc)
                if (isOcc == 1):
                    cv2.imwrite("spots_folder/generatedOccupied/" + "genImg{},{},{}".format(i, frameNum, int(startTime)) + ".jpg", img)
                elif (isOcc == 0):
                    cv2.imwrite("spots_folder/generatedEmpty/" + "genImg{},{},{}".format(i, frameNum, int(startTime)) + ".jpg", img)

    initial = False

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (15, 15), 0)
    # #ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
    #
    # # find contours
    # _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # #get viable contours
    # validCont = []
    # biggest = None
    # max_area = 0
    # for i in contours:
    #     area = cv2.contourArea(i)
    #     if area > 1000 and area < 10000:
    #         peri = cv2.arcLength(i, True)
    #         approx = cv2.approxPolyDP(i, 0.02 * peri, True)
    #         validCont.append(approx)
    #         if area > max_area and len(approx) == 4:
    #             biggest = approx
    #             max_area = area
    #
    #
    # # cv2.drawContours(frame, validCont, -1, (0, 255, 0), 3)

    # Display the resulting frame
    if(auto==False):
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.imshow('frame',frame)
        key = cv2.waitKey(0)
        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord('w'):
            diff = np.sum(np.sum(np.sum(frame-prevFrame,0),0))
            baseline = np.sum(np.sum(np.sum(frame,0),0))
            print(diff)
            print(baseline)
            print(1.0*diff/baseline)
            prevFrame=frame
            continue

        if (key & 0xFF == ord('p')):
            t = time()
            cv2.imwrite("spots_folder/frame{}.jpg".format(frameNum),frame)
            frameNum +=1
            testImages = p.getRotateRect(coordList)
            for i,img in enumerate(testImages):
                isOcc = predict(img, thetaFinal, 32, True, True)
                if(isOcc==1):
                    cv2.imwrite("spots_folder/generatedOccupied/"+"genImg{},{},{}".format(i,frameNum,int(t))+".jpg",img)
                elif(isOcc==0):
                    cv2.imwrite("spots_folder/generatedEmpty/"+"genImg{},{},{}".format(i,frameNum,int(t))+".jpg",img)


    if(auto == True):
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(10)
        if key & 0xFF == ord('q'):
            break

        currTime = time()
        if (key & 0xFF == ord('p')) or (currTime - startTime >= timeInterval*60):
            startTime = currTime
            cv2.imwrite("spots_folder/frame{}.jpg".format(frameNum), frame)
            frameNum += 1
            testImages = p.getRotateRect(coordList)
            for i, img in enumerate(testImages):

                isOcc = predict(img, thetaFinal, 32, True, True)
                if (isOcc == 1):
                    cv2.imwrite("spots_folder/generatedOccupied/" + "genImg{},{},{}".format(i,frameNum,int(currTime))+".jpg",img)
                elif (isOcc == 0):
                    cv2.imwrite("spots_folder/generatedEmpty/" + "genImg{},{},{}".format(i,frameNum,int(currTime))+".jpg",img)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
