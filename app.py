from flask import Flask, render_template, flash, request, jsonify, redirect, url_for, session, g
from SmartParking import *
from time import sleep
from onvif import ONVIFCamera


CAMERA_POSITION_NUMBER = 3

app = Flask(__name__)
# app.config['DEBUG'] = True

@app.route('/')
def index():
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
    img_name_list = ["./static/images/labeled000.jpg", "./static/images/labeled001.jpg", "./static/images/labeled002.jpg"]
    return render_template('index.html', img_name =  img_name_list, parkingInfo = parkingInfo)


if __name__ == "__main__":
    app.run(host='0.0.0.0')