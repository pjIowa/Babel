import numpy as np


def oddBitCount(x):
    return np.count_nonzero(x)%2

def sigmoid(x):
    return 1./(1+np.exp(-x))

class LSTMParam:
    def __init__(self, numMemCells, xDim):
        self.numMemCells = numMemCells
        self.xDim = xDim
        
        concatLen = xDim+numMemCells
        a = -0.1
        b = 0.1
        
        # weights
        self.wg = np.random.rand((numMemCells, concatLen))*(b-a)+a
        self.wi = np.random.rand((numMemCells, concatLen))*(b-a)+a
        self.wf = np.random.rand((numMemCells, concatLen))*(b-a)+a
        self.wo = np.random.rand((numMemCells, concatLen))*(b-a)+a
        
        # bias
        self.bg = np.random.rand(numMemCells)*(b-a)+a
        self.bi = np.random.rand(numMemCells)*(b-a)+a
        self.bf = np.random.rand(numMemCells)*(b-a)+a
        self.bo = np.random.rand(numMemCells)*(b-a)+a
        
        # derivatives
        self.wg = np.zeros((numMemCells, concatLen))
        self.wi = np.zeros((numMemCells, concatLen))
        self.wf = np.zeros((numMemCells, concatLen))
        self.wo = np.zeros((numMemCells, concatLen))
        self.bg = np.zeros(numMemCells)
        self.bi = np.zeros(numMemCells)
        self.bf = np.zeros(numMemCells)
        self.bo = np.zeros(numMemCells)
        
        

sequenceLength = 10
m = 20
outputLength = 1
numberOfEpochs = 5000

xTrain = np.around(np.random.rand(m, sequenceLength))
yTrain = np.apply_along_axis( oddBitCount, axis=1, arr=xTrain )
print xTrain
print yTrain