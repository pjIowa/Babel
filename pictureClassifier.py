import numpy as np
import struct
import os

class PictureClassifier:
    
    def __init__(self):
        self.numTrain = 100 # training batch
        self.numHidden = 1000 # hidden nodes
        self.epsilon = 0.01 # learning rate for gradient descent
        self.reg_lambda = 0.0 # regularization strength
        
        self.expMin = -350
        self.expMax = 350
        
        # avoid scientific notation
        np.set_printoptions(suppress=True)
        np.seterr(all='raise')
        
        # relu helpers
        self.RELU = lambda x: np.maximum(0,x)
        self.derivative_RELU = lambda x: 1.*(x>0)
        
        # sigmoid helpers
        self.sigmoid = lambda x: 1./(1.+np.exp(np.clip(-x,self.expMin,self.expMax)))
        self.derivative_sigmoid = lambda x: self.sigmoid(x)/(1.+self.sigmoid(x))
        
        self.loadData()
        self.loadWeights()
        
    def loadData(self):
        with open('train-labels-idx1-ubyte', 'rb') as flbl:
            magic, self.numRawTrain = struct.unpack(">II", flbl.read(8))
            self.rawTrainLabels = np.fromfile(flbl, dtype=np.int8)
            self.numUniqueLabels = len(np.unique(self.rawTrainLabels))
        print '\nLoaded Training Labels'
    
        with open('train-images-idx3-ubyte', 'rb') as fimg:
            magic, n, self.imRows, self.imCols = struct.unpack(">IIII", fimg.read(16))
            self.rawTrainImages = np.fromfile(fimg, dtype=np.uint8).reshape(n, self.imRows*self.imCols)
        print 'Loaded Training Images'
        
        with open('t10k-labels-idx1-ubyte', 'rb') as flbl:
            magic, n = struct.unpack(">II", flbl.read(8))
            self.testLabels = np.fromfile(flbl, dtype=np.int8)
        print 'Loaded Test Labels'
        
        with open('t10k-images-idx3-ubyte', 'rb') as fimg:
            magic, self.numTest, self.imRows, self.imCols = struct.unpack(">IIII", fimg.read(16))
            self.testImages = np.fromfile(fimg, dtype=np.uint8).reshape(n, self.imRows*self.imCols)
            self.testImages = np.hstack((np.ones((self.numTest, 1)), self.testImages))
        print 'Loaded Test Images\n'
        
        permuteIndices = np.random.permutation(self.numRawTrain)[:self.numTrain]
        
        self.trainLabels = self.rawTrainLabels[permuteIndices]
        self.trainImages = np.hstack((np.ones((self.numTrain, 1)), self.rawTrainImages[permuteIndices,:]))
        
    def loadWeights(self):
        np.random.seed(0)
        self.w1 = np.random.randn(self.imRows*self.imCols+1, self.numHidden)
        self.w2 = np.random.randn(self.numHidden, self.numUniqueLabels)
    
    def predict(self, x):
        # Forward propagation
        z1 = x.dot(self.w1)
        a1 = self.sigmoid(z1)
        z2 = a1.dot(self.w2)
        z2 = np.clip(z2,self.expMin,self.expMax)
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return np.argmax(probs, axis=1)
        
    def trainCost(self):
        # Forward propagation to calculate our predictions
        z1 = self.trainImages.dot(self.w1)
        a1 = self.sigmoid(z1)
        z2 = a1.dot(self.w2)
        z2 = np.clip(z2,self.expMin,self.expMax)
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # Calculating the loss
        correct_logprobs = -np.log(probs[range(self.numTrain), self.trainLabels])
        data_loss = np.sum(correct_logprobs)
        
        # Add regulatization term to loss
        data_loss += self.reg_lambda/2 * (np.sum(np.square(self.w1)) + np.sum(np.square(self.w2)))
        
        return 1./self.numTrain * data_loss
    
    def trainAccuracy(self):
        predictions = self.predict(self.trainImages)
        return (100.0 * np.sum(np.equal(predictions,self.trainLabels)) / predictions.shape[0])
    
    def testAccuracy(self):
        predictions = self.predict(self.testImages)
        return (100.0 * np.sum(np.equal(predictions,self.testLabels)) / predictions.shape[0])
    
    def updateWeights(self):
        permuteIndices = np.random.permutation(self.numRawTrain)[:self.numTrain]
        
        self.trainLabels = self.rawTrainLabels[permuteIndices]
        self.trainImages = np.hstack((np.ones((self.numTrain, 1)), self.rawTrainImages[permuteIndices,:]))
        
        # Forward propagation
        z1 = self.trainImages.dot(self.w1)
        a1 = self.sigmoid(z1)
        z2 = a1.dot(self.w2)
        z2 = np.clip(z2,self.expMin,self.expMax)
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # Backpropagation
        probs[range(self.numTrain), self.trainLabels] -= 1
        dW2 = (a1.T).dot(probs)
        delta2 = probs.dot(self.w2.T) * self.derivative_sigmoid(a1)
        dW1 = np.dot(self.trainImages.T, delta2)
 
        # Add regularization terms
        dW2 += self.reg_lambda * self.w2
        dW1 += self.reg_lambda * self.w1
 
        # Gradient descent parameter update
        self.w1 += -self.epsilon * dW1
        self.w2 += -self.epsilon * dW2
        
# NEXT: add convolutions

c = PictureClassifier()
numIterations = 2000
for i in range(numIterations):
    if i%200 == 0:
        trainAccuracy = c.trainAccuracy()
        print "Step ", i
        print "Training Loss: \t", c.trainCost()
        print "Training Accuracy: \t%.1f%%" % trainAccuracy
        
    c.updateWeights()
        
print "Final Training Loss: \t", c.trainCost()
print "Final Training Accuracy: \t%.1f%%" % c.trainAccuracy()
print "Final Test Accuracy: \t%.1f%%\n" % c.testAccuracy()