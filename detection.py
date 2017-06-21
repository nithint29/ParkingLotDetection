import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
import itertools

from dataset import loadFolder

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

''' SVM class model'''
class SVM(StatModel):
    def __init__(self, C = 1.0, gamma = 0.5, kernel = cv2.ml.SVM_LINEAR, type = cv2.ml.SVM_C_SVC, file_path = None):
        if(file_path != None):
            self.model = cv2.ml.SVM_load(file_path)
        else:
            self.model = cv2.ml.SVM_create()
            self.model.setGamma(gamma)
            self.model.setC(C)
            self.model.setKernel(kernel)
            self.model.setType(type)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()

''' DCT for feature '''
def getDCTFeature(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 1 chan, grayscale!
    imf = np.float32(img_gray)/255.0  # float conversion/scale
    dct = cv2.dct(imf)           # the dct
    (r, c) =  dct.shape
    new_dct = dct[0:r, 0:c]
    dct_vector = new_dct.flatten()
   #  img_dct = np.uint8(dst)*255.0    # convert back
   #  plt.imshow(img_dct, cmap = 'gray', interpolation = 'bicubic')
   # # plt.imshow(img_dct, interpolation = 'bicubic')
   #  plt.xticks([]), plt.yticks([])
   #  plt.show()
    return dct_vector

''' return dct feature list for input image list'''
def getDCTFeatureList(img_list):
    dct_feature_list = []
    for img in img_list:
        dct_feature = getDCTFeature(img)
        dct_feature_list.append(dct_feature)
    return dct_feature_list

''' histogram for feature '''
def getHistFeature(img, usecolor = False,  bins = 256):
    maxRange = 256
    if(usecolor):
        #split the bgr channels for image
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]
        b_hist = cv2.calcHist([b], [0], None, [bins], [0, maxRange])
        g_hist = cv2.calcHist([g], [0], None, [bins], [0, maxRange])
        r_hist = cv2.calcHist([r], [0], None, [bins], [0, maxRange])
        imghist = np.concatenate((b_hist.flatten(), g_hist.flatten(), r_hist.flatten()))
    else:
        imghist = cv2.calcHist([img], [0], None, [bins], [0, maxRange])
        imghist.flatten()

    return imghist

''' return dct feature list for input image list'''
def getHistFeatureList(img_list, usecolor = False,  bins = 256):
    hist_feature_list = []
    for img in img_list:
        hist_feature = getHistFeature(img, usecolor, bins)
        hist_feature_list.append(hist_feature)
    return hist_feature_list

''' evaluate the model '''
def evaluateModel(model, testdata, testlabel):

    resp = model.predict(testdata)
    err = (testlabel != resp).mean()
    print('error: %.2f %%' % (err*100))

    confusion = np.zeros((2, 2), np.int32)
    for i, j in zip(testlabel, resp):
        confusion[i, int(j)] += 1
    print('confusion matrix:')
    print(confusion)
    return err


def main():
    empty_list = loadFolder("dataset_resize/empty")
    occupied_list = loadFolder("dataset_resize/occupied")

    # we select 75 image in each image list as training dataset
    # use left 75 images in each image list as testing dataset
    # then we can exchange the traing and testing dataset, compute accuracy again
    train_labels = np.zeros(150, dtype=np.int)
    train_labels[75:150] = np.ones(75)
    test_labels = np.zeros(150, dtype=np.int)
    test_labels[75:150] = np.ones(75)

    # DCT features
    # empty_features = getDCTFeatureList(empty_list)
    # occupied_features = getDCTFeatureList(occupied_list)

    svm_kernel = [
	("SVM_LINEAR", cv2.ml.SVM_LINEAR)]
    #("SVM_RBF", cv2.ml.SVM_RBF),
	#("SVM_POLY", cv2.ml.SVM_POLY) ]
    svm_C = np.arange(1.0, 3.0, 1)
    svm_gamma = [5.383]
    colors = [True, False]
    binVals = [16,64,256]
    combs = list(itertools.product(svm_kernel, svm_C, svm_gamma, colors, binVals))

    bestParameter = [None] * 5
    bestFeatures = [None] * 150
    bestModel = SVM()
    min = 1.0

    for comb in combs:
        name, kernel = comb[0]
        C = comb[1]
        gamma = comb[2]
        usecolor = comb[3]
        bins = comb[4]

        # Hist features
        empty_features = getHistFeatureList(empty_list, usecolor, bins)
        occupied_features = getHistFeatureList(occupied_list, usecolor, bins)

        trainset = [None]  * 150
        testset = [None] * 150
        trainset[0:75] = empty_features[0:75]
        trainset[75:150] = occupied_features[0:75]
        testset[0:75] = empty_features[75:150]
        testset[75:150] = occupied_features[75:150]
        trainset = np.asarray(trainset)
        testset = np.asarray(testset)

        #build SVM model
        model = SVM(C, gamma, kernel)
        model.train(trainset, train_labels)
        err = evaluateModel(model, testset, test_labels)

        # model.train(testset, test_labels)
        # err = evaluateModel(model, trainset, train_labels)

        print( "SVM kernel" + name + ", SVM C = " + str(C) + ", SVM gamma = " + str(gamma))
        print("use color = " + str(usecolor) + ", binVal = " + str(bins))
        print ("Error rate: " + str(err))

        if err < min:
            bestParameter[0] = name
            bestParameter[1] = C
            bestParameter[2] = gamma
            bestParameter[3] = usecolor
            bestParameter[4] = bins
            bestModel = model
            bestFeatures = testset
            min = err

    print("The best parameter is: ")
    print( "SVM kernel: " + bestParameter[0] + ", SVM C = " + str(bestParameter[1]) + ", SVM gamma = " + str(bestParameter[2]))
    print("use color = " + str(bestParameter[3]) + ", binVal = " + str(bestParameter[4]))
    print ("Error rate: " + str(min))

    # bestModel.save("svm_model1.xml")
    #
    # print("load model")
    # svmload = SVM(file_path="svm_model1.xml")
    # err = evaluateModel(svmload, bestFeatures, test_labels)
    # print "error rate:" + str(err)



if __name__ == '__main__':
    main()