import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import glob
from sklearn import svm
import pickle

def readFromDict(imageDict):
    for i in imageDict:
        cv2.imshow("lot"+str(i),imageDict[i][1])
        cv2.waitKey(0)
    return

#Generates list of images from folder (resized to 400 X 400)
def readFromFolder(folder):
    img_list = []
    for image in os.listdir(folder):

        img = cv2.imread(os.path.join(folder,image))
        img = cv2.resize(img, (400,400), interpolation=cv2.INTER_AREA)
        if image is not None:
            img_list.append(img)
            # print( np.sum([img[:,:,0],img[:,:,1],img[:,:,2]])/(3*400*400) )
    return img_list

def loadFolder(folderPath,resizeBlur=True):
    img_list = []

    for files in glob.glob(folderPath + "/*.jpg"):
        img = cv2.imread(files)
        if(resizeBlur==True):
            img = cv2.GaussianBlur(img, (15, 15), 0)
            img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_AREA)
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        img_list.append(img)

    return img_list

''' read images from a directory, resize images and save '''
def loadResizeSave(srcPath, savePath):
    for i, imgs in enumerate(glob.glob(srcPath + "/*.jpg")):
        img = cv2.imread(imgs)
        img_resize = cv2.resize(img, (400, 400), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(savePath + "/" + str(i + 1) + ".jpg", img_resize)

    return

#run through image color or grayscale histograms
def displayHist(imgList,isColor,empty,occupied,method):
    for img in imgList:
        # img[:,:,0] = cv2.equalizeHist(img[:,:,0])
        # img[:, :, 1] = cv2.equalizeHist(img[:,:,1])
        # img[:, :, 2] = cv2.equalizeHist(img[:,:,2])
        if(isColor == False):
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        plt.subplot(221), plt.imshow(img, 'gray')
        plt.subplot(222),

        if(isColor):
            colors = ('b', 'g', 'r')
        else:
            colors = ('g')
        for i, col in enumerate(colors):
            #img = cv2.equalizeHist(img)
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
            plt.xlim([0, 256])
            # coeff = cv2.compareHist(hist,compHist,cv2.HISTCMP_BHATTACHARYYA)
            # if((coeff-.5)<0.15):
            #     print"Empty: ",coeff
            # else:
            #     print "Occupied: ", coeff
        #if(isColor == False):
        #print histClassify(img,empty,occupied,method)
        plt.show()


#for grayscale image classification
def histClassify(image,empty,occupied,method = cv2.HISTCMP_CORREL,useColor = False,bins = 256):
    if(useColor):
        emptyHist = cv2.calcHist([empty], [0,1,2], None, [bins,bins,bins], [0, 256,0,256,0,256])
        occupiedHist = cv2.calcHist([occupied], [0,1,2], None, [bins,bins,bins], [0, 256,0,256,0,256])
        imageHist = cv2.calcHist([image], [0,1,2], None, [bins,bins,bins], [0, 256,0,256,0,256])
    else:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        empty = cv2.cvtColor(empty, cv2.COLOR_BGR2GRAY)
        occupied = cv2.cvtColor(occupied, cv2.COLOR_BGR2GRAY)
        emptyHist = cv2.calcHist([empty], [0], None, [bins], [0, 256])
        occupiedHist = cv2.calcHist([occupied], [0], None, [bins], [0, 256])
        imageHist = cv2.calcHist([image], [0], None, [bins], [0, 256])
    coeffEmpty = cv2.compareHist(imageHist,emptyHist,method)
    coeffOccupied = cv2.compareHist(imageHist, occupiedHist, method)
    coeffSelf = cv2.compareHist(imageHist, imageHist, method)

    if(abs(coeffSelf-coeffEmpty)<abs(coeffSelf-coeffOccupied)):
        #print "empty: " +str(coeffSelf)+", " + str(abs(coeffSelf-coeffEmpty))+", "+ str(abs(coeffSelf-coeffOccupied))
        return "empty"
    else:
        #print "occupied: "+str(coeffSelf)+", "+str(abs(coeffSelf-coeffEmpty))+", "+ str(abs(coeffSelf-coeffOccupied))
        return "occupied"

def histClassify2(image, empty, occupied, method=cv2.HISTCMP_CORREL, useColor=False, bins=256):
    emptyHist = []
    occupiedHist = []
    if (useColor):
        for i in range(len(empty)):
            emptyHist.append(cv2.calcHist([empty[i]], [0, 1, 2], None, [bins, bins, bins], [0, bins, 0, bins, 0, bins]))
            occupiedHist.append(cv2.calcHist([occupied][i], [0, 1, 2], None, [bins, bins, bins], [0, bins, 0, bins, 0, bins]))
            imageHist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, bins, 0, bins, 0, bins])

    else:
        for i in range(len(empty)):
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # empty[i] = cv2.cvtColor(empty[i], cv2.COLOR_BGR2GRAY)
            # occupied[i] = cv2.cvtColor(occupied[i], cv2.COLOR_BGR2GRAY)
            emptyHist.append(cv2.calcHist([empty[i]], [0], None, [bins], [0, bins]))
            occupiedHist.append(cv2.calcHist([occupied[i]], [0], None, [bins], [0, bins]))
            imageHist = cv2.calcHist([image], [0], None, [bins], [0, bins])
        coeffSelf = cv2.compareHist(imageHist, imageHist, method)

    coeffEmpty = []
    coeffOccupied = []
    voteEmpty = 0
    voteOccupied = 0
    for i in range(len(empty)):
        coeffEmpty.append(cv2.compareHist(imageHist, emptyHist[i], method))
        coeffOccupied.append(cv2.compareHist(imageHist, occupiedHist[i], method))
        if (abs(coeffSelf - coeffEmpty[i]) < abs(coeffSelf - coeffOccupied[i])):
            # print "empty: " +str(coeffSelf)+", " + str(abs(coeffSelf-coeffEmpty))+", "+ str(abs(coeffSelf-coeffOccupied))
            voteEmpty+=1
        else:
            # print "occupied: "+str(coeffSelf)+", "+str(abs(coeffSelf-coeffEmpty))+", "+ str(abs(coeffSelf-coeffOccupied))
            voteOccupied+=1

    if(voteEmpty>voteOccupied):
        return "empty"
    else:
        return "occupied"


#Computes pixel averages either by channel or overall - use later for logistic regression (try grayscale,color,HSV)
def computePixels(imgList):
    pixels = []
    for image in imgList:
        #pixels.append([ np.sum(image[:,:,0])/(400*400), np.sum(image[:,:,1])/(400*400), np.sum(image[:,:,2])/(400*400) ])
        pixels.append(np.sum([image[:,:,0],image[:,:,1],image[:,:,2]])/(3*400*400))
    print(pixels)
    return pixels

#computes the cost and gradient given a theta and data - USE NUMPY ARRAYS, theta,y are columns
def costfunction(X,y,theta,lam = 0):
    n = len(X)
    if(n != len(y)):
        print("Size of X and y does not match")
        return None
    cost = (1.0/n)*np.sum(np.dot(-1.0*y.T,np.log10(sigmoid( (np.dot(1.0*X,theta))/(400.0*400) ))) - np.dot((1.0-y).T, np.log10(1.0-sigmoid( np.dot(1.0*X,theta)/(400.0*400) )) ) ,axis=0)
    gradient = np.dot((1.0/n)*(X.T),sigmoid(X.dot(theta))-y)
    gradient[1:] += (lam/X.shape[0])*theta[1:]
    return [cost,gradient]

def sigmoid(x):
    return (1.0 / (1 + np.exp(-1.0 * x)))

#trains on input array data and returns a column of final theta values
def logisticTrain(X,y,theta,alpha = 0.1,iter = 500,lam=0):
    for i in range(iter):
        cost, gradient = costfunction(X, y, theta,lam)
        #print cost
        theta = theta - alpha*gradient

    return theta

#generate training data from image list of color images
def createData(imgList,bins = 64,useColor = False,multiDim = False,hists=True,usePixels=True,imgSize=100):
    trainData = []
    colors = ("b", "g", "r")

    if(useColor==False and multiDim == True):
        print("Can not use multidimensional histograms without color, using grayscale only instead")

    for i,img in enumerate(imgList):
        if(hists == False):
            trainData.append(img.flatten())

        elif(useColor and multiDim ):
            trainData.append(np.array(cv2.calcHist([img], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])).flatten())

        elif(useColor == True and multiDim == False):
            temp = []
            for i,color in enumerate(colors):
                temp.append(np.array(cv2.calcHist([img],[i],None,[bins],[0,256])))
            trainData.append(np.array(temp).flatten())

        else:
            if(len(img[0][0]) !=1):
                img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            trainData.append(np.array(cv2.calcHist([img], [0], None, [bins], [0, 256])).flatten())

        if (usePixels):
            data = trainData.pop()
            img_data = cv2.resize(img, (imgSize, imgSize))
            data = np.concatenate((data.flatten(), img_data.flatten()))
            trainData.append(data)

    return trainData

#takes training data from input folder and outputs the resulting theta (with one extra dimension)
#set trainNum = -1 to use whole folder
def trainOnFolder(emptyFolder,occupiedFolder,trainNum,bins,color,multi,alpha = 0.1,iters = 50,hists = True,lam=0,usePixels=True,
                  imgSize=100):
    # best 100,true,true,16
    emptySet = loadFolder(emptyFolder)
    occupiedSet = loadFolder(occupiedFolder)

    trainNumE = trainNum
    trainNumO = trainNum

    if(trainNum ==-1):
        trainNumE = len(emptySet)
        trainNumO = len(occupiedSet)

    trainX1 = createData(emptySet[0:trainNumE], bins, color, multi,hists=hists,usePixels=usePixels,imgSize=imgSize)
    trainX2 = createData(occupiedSet[0:trainNumO], bins, color, multi,hists=hists,usePixels=usePixels,imgSize=imgSize)
    trainX = trainX1 + trainX2
    print(np.shape(trainX))

    X = np.ones((np.shape(trainX)[0], np.shape(trainX)[1] + 1))
    X[:, 1:X.shape[1]] = trainX
    print(np.shape(X))
    theta = np.zeros((X.shape[1], 1))
    print(np.shape(theta))
    y1 = np.zeros((trainNumE, 1))
    print(len(y1))
    y2 = np.ones((trainNumO, 1))
    print(len(y2))
    y = np.concatenate((y1, y2), axis=0)
    print(np.shape(y))

    answer = logisticTrain(X, y, theta, alpha, iters,lam)

    return answer


#classifys input image given a trained theta and raw image
def predict(img,theta,bins,useColor,multiDim,hists=True,usePixels=True,imgSize=100):
    colors = ("b", "g", "r")
    img = cv2.GaussianBlur(img, (15, 15), 0)
    img = cv2.resize(img, (400,400), interpolation=cv2.INTER_AREA)

    if(hists==False):
        X = img.flatten()

    elif (useColor == False and multiDim == True):
        print("Can not use multidimensional histograms without color, using grayscale only instead")

    elif (useColor and multiDim):
        X = (np.array(cv2.calcHist([img], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])).flatten())

    elif (useColor == True and multiDim == False):
        temp = []
        for i, color in enumerate(colors):
            temp.append(np.array(cv2.calcHist([img], [i], None, [bins], [0, 256])))
        X = (np.array(temp).flatten())

    else:
        if (len(img[0][0]) != 1):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X = (np.array(cv2.calcHist([img], [0], None, [bins], [0, 256])).flatten())

    if (usePixels):
        img_data = cv2.resize(img, (imgSize, imgSize))
        X = np.concatenate((X, img_data.flatten()))

    Xbias = np.ones(np.shape(X)[0]+1)
    Xbias[1:(np.shape(Xbias)[0])] = X
    return (1 if (sigmoid(np.dot(Xbias,theta))>0.5) else 0)

def predictSet(imgList,theta,bins,useColor,multiDim,hists=True,usePixels=True,imgSize=100):
    preds = []
    for img in imgList:
        preds.append(predict(img,theta,bins,useColor,multiDim,hists=True,usePixels=usePixels,imgSize=imgSize))
    return preds



if __name__ == "__main__":

    # thetaFinal = trainOnFoloder("rawdataset/empty","rawdataset/occupied",150,32,True,True)
    # # img = cv2.imread("mydata/occupied/spot_2.jpg")
    # # img2 = cv2.imread("mydata/empty/137.jpg")
    # #print(predict(img,thetaFinal,16,True,True))
    # # print(predict(img2, thetaFinal, 16, True, True))
    #
    # testFolder = loadFolder("spots_folder")
    #
    # for img in testFolder:
    #     print predict(img,thetaFinal,32,True,True)




    # regression
    emptySet = loadFolder("rawdataset/empty")
    occupiedSet = loadFolder("rawdataset/occupied")
    print(len(emptySet))
    print(len(occupiedSet))

    #best 100,true,true,16
    trainNum = 200
    color = True
    multi = True
    bins = 32

    trainX1 = createData(emptySet[0:trainNum],bins,color,multi)
    trainX2 = createData(occupiedSet[0:trainNum],bins,color,multi)
    trainX = trainX1+trainX2
    print(np.shape(trainX))

    X = np.ones((np.shape(trainX)[0],np.shape(trainX)[1]+1))
    X[:,1:X.shape[1]] = trainX
    X = X+0.001*np.ones(X.shape)
    print(np.shape(X))
    theta = np.zeros((X.shape[1],1))
    print(np.shape(theta))
    y1 = np.zeros((len(X)/2,1))
    y2 = np.ones((len(X)/2,1))
    y = np.concatenate((y1,y2),axis = 0)
    print(np.shape(y))

    print("cost: ")
    print(costfunction(X,y,theta)[0])
    answer = logisticTrain(X,y,theta,0.1,50,10)
    print("theta final: ")
    print(answer)

    testX1 = createData(emptySet[trainNum:],bins,color,multi)
    testX2 = createData(occupiedSet[trainNum:],bins,color,multi)
    testX = testX1+testX2
    Xtest = np.ones((np.shape(testX)[0], np.shape(testX)[1] + 1))
    Xtest[:, 1:X.shape[1]] = testX
    print(np.shape(testX))
    y1test = np.zeros((len(testX1),1))
    y2test = np.ones((len(testX2),1))
    ytest = np.concatenate((y1test,y2test),axis = 0)
    print(np.shape(ytest))

    correctEmpty = 0
    correctOcc = 0
    for i,img in enumerate(Xtest):
        if(sigmoid(np.dot(img,answer))<0.5 and ytest[i]==0):
            correctEmpty+=1
        elif(sigmoid(np.dot(img,answer))>0.5 and ytest[i]==1):
            correctOcc+=1

    print(len(testX1),correctEmpty,len(testX2),correctOcc)
    print(1.0*correctEmpty/(len(testX1)))
    print(1.0*correctOcc/(len(testX2)))
    print("\n")

    #Testing on spots_folder
    #thetaFinal = trainOnFolder("rawdataset/empty", "rawdataset/occupied", -1, 32, True, True, lam=100,usePixels=True,iters=100)
    #testFolder = loadFolder("spots_folder", False)
    #Save theta values to file
    # output = open('LR.pkl','wb')
    # pickle.dump(thetaFinal,output)
    # output.close()
    read = open('LR.pkl', 'rb')
    thetaFinal = pickle.load(read)
    read.close()
    testEmpty = np.array(loadFolder("spots_folder/generatedEmpty",False))
    testOcc = np.array(loadFolder("spots_folder/generatedOccupied",False))
    print("Empty correct: {}".format(np.sum(np.array(predictSet(testEmpty,thetaFinal, 32, True, True,usePixels=True))==0)/(1.0*len(testEmpty))))
    print("Occupied correct: {} \n".format(np.sum(predictSet(testOcc, thetaFinal, 32, True, True,usePixels=True)) / (1.0 * len(testOcc))))



    print("\nScikit Learn Stuff")
    print("SVM:")
    emptySet = np.array(emptySet)
    occupiedSet = np.array(occupiedSet)
    rng = np.random.RandomState(5)
    X = createData(np.concatenate((emptySet,occupiedSet),axis=0),32,color,multi,hists=True,usePixels=True)
    X = np.array(X)
    y = np.concatenate((np.zeros(len(emptySet)),np.ones(len(occupiedSet))))
    print(len(emptySet))
    print(len(occupiedSet))

    #randomize data order
    ind = np.floor(rng.rand(len(X))*len(X)).astype(int)
    X = X[ind]
    y = y[ind]

    print(np.shape(X))
    print(y.shape)
    clf = svm.SVC(C=1,kernel='linear')
    print(clf.fit(X[0:200],y[0:200]).score(X[200:],y[200:]))

    k = 3
    Xfolds = np.array_split(X,k)
    yfolds = np.array_split(y,k)
    scores = []


    for i in range(k):
        Xtrain = list(Xfolds)
        Xtest = Xtrain.pop(i)
        ytrain = list(yfolds)
        ytest = ytrain.pop(i)

        Xtrain = np.concatenate(Xtrain)
        ytrain = np.concatenate(ytrain)

        scores.append(clf.fit(Xtrain,ytrain).score(Xtest,ytest))

    print(scores)



    # # images = loadFolder("dataset/occupied")
    # # computePixels(images)
    # # displayHist(images,True,None,None,None)
    # emptySet = loadFolder("dataset/empty")
    # occupiedSet = loadFolder("dataset/occupied")
    #
    # #analyze using histograms
    # emptyLot = cv2.imread("dataset/empty/26.jpg")
    # emptyLot = cv2.resize(emptyLot, (400,400), interpolation=cv2.INTER_AREA)
    #
    # occupiedLot = cv2.imread("dataset/occupied/6.jpg")
    # occupiedLot = cv2.resize(occupiedLot, (400,400), interpolation=cv2.INTER_AREA)
    #
    # # emptyLot = emptySet[6:7]
    # # occupiedLot = occupiedSet[6:7]
    #
    # OPENCV_METHODS = [
    #     ("Correlation", cv2.HISTCMP_CORREL),
    #     ("Chi-Squared", cv2.HISTCMP_CHISQR),
    #     ("Intersection", cv2.HISTCMP_INTERSECT),
    #     ("Hellinger", cv2.HISTCMP_HELLINGER),
    #     ("Chi-Squared Alt",cv2.HISTCMP_CHISQR_ALT) ]
    #
    # #displayHist(emptySet,True,emptyLot,occupiedLot,cv2.HISTCMP_BHATTACHARYYA)
    #
    # binVals = [16,32,64,128]
    # color = [False]
    # combs = list(itertools.product(OPENCV_METHODS,binVals,color))
    #
    # methodAccuracies = []
    # for comb in combs:
    #     name,method = comb[0]
    #     binNum = comb[1]
    #     color = comb[2]
    #     numCorrectE =0
    #     for image in emptySet:
    #         if(histClassify(image,emptyLot,occupiedLot,method,color,binNum)=="empty"):
    #             numCorrectE = numCorrectE+1
    #     print name + ": Empty Space Accuracy = "+ str(numCorrectE*1.0/len(emptySet))
    #
    #     numCorrectO = 0
    #     for image in occupiedSet:
    #         if(histClassify(image,emptyLot,occupiedLot,method,color,binNum)=="occupied"):
    #             numCorrectO = numCorrectO+1
    #     print name + ": Occupied Space Accuracy = "+ str(numCorrectO*1.0/len(occupiedSet))
    #
    #     print "Overall Accuracy: " + str((numCorrectE*1.0/len(emptySet)+numCorrectO*1.0/len(occupiedSet))/2.0)
    #     methodAccuracies.append((comb,[numCorrectE*1.0/len(emptySet),numCorrectO*1.0/len(occupiedSet)]))
    #
    # bestAcc = 0
    # bestComb = None
    # for value in methodAccuracies:
    #     accuracy = (value[1][0]+value[1][1])/2.0
    #     if(accuracy>bestAcc):
    #         bestAcc = accuracy
    #         bestComb = value
    #
    # print bestComb
