import numpy as np
import struct
import os

class PictureClassifier:
    
    def __init__(self):
        self.loadData()
        self.loadWeights()
        np.set_printoptions(suppress=True)
        self.sigmoid = lambda x: 1.0/(1.0 + np.exp(np.clip(-x, -709, 709)))
        
    def loadData(self):
        with open('train-labels-idx1-ubyte', 'rb') as flbl:
            magic, self.numTotalExamples = struct.unpack(">II", flbl.read(8))
            labels = np.fromfile(flbl, dtype=np.int8)
            allLabelNames = np.unique(labels)
            self.labels = np.array([[1.0*(elem==j) for j in allLabelNames] for elem in labels])
            self.numUniqueLabels = len(allLabelNames)
    
        with open('train-images-idx3-ubyte', 'rb') as fimg:
            magic, self.numTotalExamples, self.imRows, self.imCols = struct.unpack(">IIII", fimg.read(16))
            self.images = np.fromfile(fimg, dtype=np.uint8).reshape(len(labels), self.imRows*self.imCols)
        
        self.numTest = 1000
        self.numTrain = 1000
        permuteIndices = np.random.permutation(self.numTotalExamples-self.numTest)[:self.numTrain]
        self.trainLabels = self.labels[permuteIndices]
        self.trainImages = np.hstack((np.ones((self.numTrain, 1)), self.images[permuteIndices,:]))
        self.testLabels = self.labels[-self.numTest:]
        self.testImages = np.hstack((np.ones((self.numTest, 1)), self.images[-self.numTest:,:]))
        
    def loadWeights(self):
        np.random.seed(0)
        self.weights = np.random.randn(self.imRows*self.imCols+1, self.numUniqueLabels)
    
    def predict(self, x):
        predictions = self.sigmoid(np.dot(x, self.weights))
        predictions[predictions < 10**-10] = 10**-10
        predictions[predictions >= 1] = 1-10**-10
        return predictions
        
    def trainCost(self):
        predictions = self.predict(self.trainImages)
        positiveAccuracy = self.trainLabels*np.log10(predictions)
        negativeAccuracy = (1-self.trainLabels)*np.log10(1-predictions)
        return -1.0/self.numTrain*np.sum(positiveAccuracy+negativeAccuracy)
    
    def trainAccuracy(self):
        predictions = self.predict(self.trainImages)
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(self.trainLabels, 1)) / predictions.shape[0])
    
    def testAccuracy(self):
        predictions = self.predict(self.testImages)
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(self.testLabels, 1)) / predictions.shape[0])
    
    def updateWeights(self):
        permuteIndices = np.random.permutation(self.numTotalExamples-self.numTest)[:self.numTrain]
        self.trainLabels = self.labels[permuteIndices]
        self.trainImages = np.hstack((np.ones((self.numTrain, 1)), self.images[permuteIndices,:]))
        
        predictError = (self.predict(self.trainImages)-self.trainLabels)**2
        learnRate = 0.0000001
        self.weights -= learnRate/self.numTrain * np.sum(np.dot((predictError).T, self.trainImages).T)
        
# NEXT: try cross entropy error
# NEXT: create more layers and convolutions

c = PictureClassifier()
for i in range(1000):
    c.updateWeights()
    if i%100 == 0:
        print "Step ", i
        print "Training Loss: \t", c.trainCost()
        print "Training Accuracy: \t%.1f%%" % c.trainAccuracy()
        print "Test Accuracy: \t%.1f%%\n" % c.testAccuracy()