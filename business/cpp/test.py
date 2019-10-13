import pandas as pd
import numpy as np
import random
import math
import pylab as pl
import numpy as np
from scipy import sparse
from matplotlib.colors import ListedColormap
import python_modules.ML as ML

#Train data generator
def generateData (numberOfClassEl, numberOfClasses):
    data = []
    for classNum in range(numberOfClasses):
        #Choose random center of 2-dimensional gaussian
        centerX, centerY = random.random()*10.0, random.random()*10.0
        #Choose numberOfClassEl random nodes with RMS=0.5
        for rowNum in range(numberOfClassEl):
            data.append(ML.TrainElement(random.gauss(centerX,0.5), random.gauss(centerY,0.5), classNum))
    return data

def splitTrainTest (data, testPercent):
    trainData = []
    testData  = []
    testDataWithLabels = []
    for row in data:
        if random.random() < testPercent:
            testData.append(ML.TestElement(row))
            testDataWithLabels.append(row)
        else:
            trainData.append(row)
    return trainData, testData, testDataWithLabels

def calculateAccuracy (nClasses, nItemsInClass, k, testPercent):
    data = generateData (nItemsInClass, nClasses)
    trainData, testData, testDataWithLabels = splitTrainTest (data, testPercent)
    testDataLabels = ML.classifyKnn (trainData, testData, k, nClasses)
    print("Accuracy: ", sum([int(testDataLabels[i]==testDataWithLabels[i].class_number) for i in range(len(testDataWithLabels))]) / float(len(testDataWithLabels)))



# calculateAccuracy(3,20,1,0.3)


from sklearn.cross_validation import KFold
