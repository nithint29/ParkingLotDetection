import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
import os

HEIGHT = 400
WIDTH = 400

''' read images from a directory '''
def loadFolder(folderPath):
    img_list = []
    for files in glob.glob(folderPath + "/*.jpg"):
        img = cv2.imread(files, cv2.IMREAD_COLOR)
        img_list.append(img)
    return img_list


''' read images from a directory, resize images and save '''
def loadResizeSave(srcPath, savePath):
    for i, imgs in enumerate(glob.glob(srcPath + "/*.jpg")):
        img = cv2.imread(imgs)
        img_resize = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(savePath + "/" + "{:03d}".format(i) + ".jpg", img_resize)
    return

# resize_empty_list = loadResizeSave("dataset/empty","dataset_resize/empty")
# resize_occupied_list = loadResizeSave("dataset/occupied","dataset_resize/occupied")

