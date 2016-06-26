import numpy as np
import struct
import os

class PictureClassifier:
    
    def __init__(self):
        self.loadData()
        self.f = lambda x: 1.0/(1.0 + np.exp(np.clip(-x, -709, 709)))
        
    def loadData(self):
        with open('train-labels-idx1-ubyte', 'rb') as flbl:
            magic, self.numExamples = struct.unpack(">II", flbl.read(8))
            labels = np.fromfile(flbl, dtype=np.int8)
            allLabelNames = np.unique(labels)
            labels = np.array([[1.0*(elem==j) for j in allLabelNames] for elem in labels])
            self.numUniqueLabels = len(allLabelNames)
    
        with open('train-images-idx3-ubyte', 'rb') as fimg:
            magic, num, self.imRows, self.imCols = struct.unpack(">IIII", fimg.read(16))
            images = np.fromfile(fimg, dtype=np.uint8).reshape(len(labels), self.imRows*self.imCols)
        
        numTest = 1000
        self.trainLabels = labels[:num-numTest]
        self.trainImages = images[:num-numTest,:]
        self.testLabels = labels[-numTest:]
        self.testImages = images[-numTest:,:]
        
        np.random.seed(0)
        self.weights = np.random.randn(self.imRows*self.imCols, self.numUniqueLabels)
        self.biases = np.random.randn(self.numUniqueLabels)
    
    def predict(self, x):
        predictions = self.f(np.dot(x, self.weights) + self.biases)
        predictions[predictions < 10**-10] = 10**-10
        predictions[predictions >= 1] = 1-10**-10
        return predictions
        
    def trainCost(self):
        predictions = self.predict(self.trainImages)
        positiveAccuracy = self.trainLabels*np.log10(predictions)
        negativeAccuracy = (1-self.trainLabels)*np.log10(1-predictions)
        return -1.0/self.numExamples*np.sum(positiveAccuracy+negativeAccuracy)
    
    def updateWeights(self):
        predictError = self.predict(self.trainImages)-self.trainLabels
        weightGrads = np.dot((predictError).T, self.trainImages).T
        learnRate = 0.0000001
        self.weights -= learnRate/self.numExamples * np.sum(weightGrads)
        self.biases -= learnRate/self.numExamples * np.sum(predictError)
        

c = PictureClassifier()
for i in range(4000):
    c.updateWeights()
    print "Step ", i, "\t", c.trainCost()