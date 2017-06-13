import cv2


cap = cv2.VideoCapture("rtsp://bigbrother.winlab.rutgers.edu/stream1")

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (15, 15), 0)
    #ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

    # find contours
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #get viable contours
    validCont = []
    biggest = None
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 1000 and area < 10000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            validCont.append(approx)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area


    cv2.drawContours(frame, validCont, -1, (0, 255, 0), 3)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
