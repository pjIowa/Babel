import numpy as np

def oddBitCount(x):
    return np.count_nonzero(x)%2

sequenceLength = 10
m = 20
outputLength = 1
numberOfEpochs = 5000

xTrain = np.around(np.random.rand(m, sequenceLength))
yTrain = np.apply_along_axis( oddBitCount, axis=1, arr=xTrain )
print xTrain
print yTrain