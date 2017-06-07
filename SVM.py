import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt


''' read images from a directory '''
def loadFolder(folderPath):
    img_list = []

    for files in glob.glob(folderPath + "/*.jpg"):
        file = cv2.imread(files)
        img_list.append(file)

    return img_list


if __name__ == "__main__":
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(2.67)






''' below is a sample code '''

# Set up SVM for OpenCV 3
svm = cv2.ml.SVM_create()
# Set SVM type
svm.setType(cv2.ml.SVM_C_SVC)
# Set SVM Kernel to Radial Basis Function (RBF)
svm.setKernel(cv2.ml.SVM_RBF)
# Set parameter C
svm.setC(C)
# Set parameter Gamma
svm.setGamma(gamma)

# Train SVM on training data
svm.train(trainData, cv2.ml.ROW_SAMPLE, trainLabels)

# Save trained model
svm->save("digits_svm_model.yml");

# Test on a held out test set
testResponse = svm.predict(testData)[1].ravel()
