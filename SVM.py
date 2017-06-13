import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
import os

HEIGHT = 200
WIDTH = 124

''' read images from a directory '''
def loadFolder(folderPath):
    img_list = []

    for files in glob.glob(folderPath + "/*.jpg"):
        file = cv2.imread(files)
        img_list.append(file)

    return img_list


''' read images from a directory, resize images and save '''
def loadResizeSave(srcPath, savePath):
    for i, imgs in enumerate(glob.glob(srcPath + "/*.jpg")):
        img = cv2.imread(imgs)
        img_resize = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(savePath + "/" + str(i + 1) + ".jpg", img_resize)

    return


if __name__ == "__main__":
    loadResizeSave("", "")

    # svm = cv2.ml.SVM_create()
    # svm.setType(cv2.ml.SVM_C_SVC)
    # svm.setKernel(cv2.ml.SVM_LINEAR)
    # svm.setC(2.67)






''' below is a sample code '''

# # Set up SVM for OpenCV 3
# svm = cv2.ml.SVM_create()
# # Set SVM type
# svm.setType(cv2.ml.SVM_C_SVC)
# # Set SVM Kernel to Radial Basis Function (RBF)
# svm.setKernel(cv2.ml.SVM_RBF)
# # Set parameter C
# svm.setC(C)
# # Set parameter Gamma
# svm.setGamma(gamma)
#
# # Train SVM on training data
# svm.train(trainData, cv2.ml.ROW_SAMPLE, trainLabels)
#
# # Save trained model
# svm->save("digits_svm_model.yml");
#
# # Test on a held out test set
# testResponse = svm.predict(testData)[1].ravel()
