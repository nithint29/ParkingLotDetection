import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt

from dataset import loadFolder

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)  # Known bug: https://github.com/opencv/opencv/issues/4969
    def save(self, fn):
        self.model.save(fn)

''' SVM class model'''
class SVM(StatModel):
    def __init__(self, C = 1.0, gamma = 0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_LINEAR)
        self.model.setType(cv2.ml.SVM_C_SVC)

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
        imghist = cv2.calcHist([img], [0,1,2], None, [bins,bins,bins], [0, maxRange,0,maxRange,0,maxRange])
    else:
        imghist = cv2.calcHist([img], [0], None, [bins], [0, maxRange])


    imghist = imghist.flatten()

    return imghist

''' return dct feature list for input image list'''
def getHistFeatureList(img_list):
    hist_feature_list = []
    for img in img_list:
        hist_feature = getHistFeature(img)
        hist_feature_list.append(hist_feature)
    return hist_feature_list

''' evaluate the model '''
def evaluateModel(model, testdata, testlabel):
    # tp_count = tn_count = fp_count = fn_count = 0
    # correct_count = 0
    # wrong_count = 0

    # for i, data in enumerate(testdata):
    #     model.predict(data)

    resp = model.predict(testdata)
    err = (testlabel != resp).mean()
    print('error: %.2f %%' % (err*100))

    confusion = np.zeros((2, 2), np.int32)
    for i, j in zip(testlabel, resp):
        confusion[i, int(j)] += 1
    print('confusion matrix:')
    print(confusion)




def main():
    empty_list = loadFolder("dataset_resize/empty")
    occupied_list = loadFolder("dataset_resize/occupied")

    # DCT features
    # empty_features = getDCTFeatureList(empty_list)
    # occupied_features = getDCTFeatureList(occupied_list)

    # Hist features
    empty_features = getHistFeatureList(empty_list)
    occupied_features = getHistFeatureList(occupied_list)

    # we select 75 image in each image list as training dataset
    # use left 75 images in each image list as testing dataset
    # then we can exchange the traing and testing dataset, compute accuracy again
    trainset = [None]  * 150
    testset = [None] * 150
    trainset[0:75] = empty_features[0:75]
    trainset[75:150] = occupied_features[0:75]
    testset[0:75] = empty_features[75:150]
    testset[75:150] = occupied_features[75:150]
    trainset = np.asarray(trainset)
    testset = np.asarray(testset)

    train_labels = np.zeros(150, dtype=np.int)
    train_labels[75:150] = np.ones(75)
    test_labels = np.zeros(150, dtype=np.int)
    test_labels[75:150] = np.ones(75)

    #build SVM model
    model = SVM(C=2.67, gamma=5.383)
    # model.train(trainset, train_labels)
    # evaluateModel(model, testset, test_labels)

    model.train(testset, test_labels)
    evaluateModel(model, trainset, train_labels)



if __name__ == '__main__':
    main()