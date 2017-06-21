import numpy as np
import cv2
from matplotlib import pyplot as plt

HEIGHT = 400
WIDTH = 400

''' extract spot mannuly'''
def getSpotsCoordiantesFromImage(img, num_space) :
    #coordinate_lists has this format[ [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], [], [] ]
    coordinate_lists = []
    spots_index_list = []
    for i in range(num_space):
        plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])
        #we need 4 points to get rectangle
        print("Please click 4 points for parking lot in clock direction", i)
        coordinate = plt.ginput(4)
        print("clicked points coordinate are ", coordinate)
        coordinate_lists.append(coordinate)
        spots_index_list.append(i)
    plt.close()
    saveSpotsCoordinates(coordinate_lists)
    saveSpotsIndex(spots_index_list)
    return coordinate_lists

''' get rotate rectangle '''
def getRotateRect(img, cooridnate_lists):
    #warped image list is the list with warper images
    warped_img_lists = []
    #every time we process one coordinates
    for coordinate in cooridnate_lists :
        warped = four_point_transform(img, coordinate)
        warped_resize = cv2.resize(warped, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
        # plt.imshow(warped, cmap = 'gray', interpolation = 'bicubic')
        # plt.xticks([]), plt.yticks([])
        # plt.show()
        warped_img_lists.append(warped_resize)
    return warped_img_lists

''' return warped image by 4 points'''
def four_point_transform(image, coordinate):
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
        [0, maxHeight - 1]], dtype = "float32")

    #change original points list to nparray and set to the same dtype
    rect = np.asarray(coordinate).astype(dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

''' save the coordinate lists into txt file'''
def saveSpotsCoordinates(coordinate_lists):
    file = open("coordinates.txt", "w")
    for coordinate in coordinate_lists:
        for t in coordinate:
            file.write(' '.join(str(s) for s in t) + '\n')
    file.close()
    print("save coordinates successfully")

''' save the spots index lists into txt file'''
def saveSpotsIndex(spots_index_list):
    file = open("spots_index.txt", "w")
    for index in spots_index_list:
        file.write(str(index) + '\n')
    file.close()
    print("save index successfully")

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
            #every 4 points as a rectangle
            if count == 4:
                count = 0
                coordinate_lists.append(temp_list)
                temp_list = []
    print("read coordinates lists successfully")
    return coordinate_lists

''' read spot index file and return it as spot_index_list'''
def readSpotsIndex(filename):
    spots_index_list = []
    with open(filename) as file:
        for line in file:
            string_format = line.strip('\n')
            spots_index_list.append(int(string_format))
    print("read spots lists successfully")
    return spots_index_list

''' save image list with path'''
def saveImageList(img_list, save_path):
    for i, img in enumerate(img_list):
        cv2.imwrite(save_path + "/spot_" + str(i) + ".jpg", img)
    print("save N = " + str(len(img_list)) + " in path " + save_path)

def main():
    img1 = cv2.imread('view5.png', cv2.IMREAD_COLOR)
    #edit num_of_lots to determine how many
    num_of_lots = 3
    getSpotsCoordiantesFromImage(img1,num_of_lots)
    #the ith coordinate list correspond to ith spots index list
    coordinate_lists = readSpotsCoordinates("coordinates.txt")
    spots_index_lists = readSpotsIndex("spots_index.txt")
    warp_img_list = getRotateRect(img1,coordinate_lists)
    saveImageList(warp_img_list, "spots_folder")
    print(spots_index_lists)

if __name__ == '__main__':
    main()