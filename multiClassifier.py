
import numpy as np
import time

def parseData():
    csv = np.loadtxt ('iris.csv', delimiter=',', dtype = str)
    y = csv[:,-1]
    allLabelNames = sorted(set(y.flat))
    numLabels = len(allLabelNames)
    y = np.array([[1.0*(elem==j) for j in allLabelNames] for elem in y])
    x = csv[:,:-1].astype(np.float)
    return np.hstack((np.ones((x.shape[0], 1)), x)), y, numLabels

def cost(w,x,y):
    numExamples = x.shape[0]
    predictions = predict(w, x)
    positiveAccuracy = y*np.log10(predictions)
    negativeAccuracy = (1-y)*np.log10(1-predictions)
    return -1.0/numExamples*np.sum(positiveAccuracy+negativeAccuracy)

def predict(w,x):
    predictions = 1.0/(1.0+np.exp(np.dot(x, -w)))
    predictions[predictions < 10**-10] = 10**-10
    predictions[predictions >= 1] = 1-10**-10
    return predictions

def updateWeights(w,x,y):
    numExamples = x.shape[0]
    gradients = np.dot((predict(w, x)-y).T, x).T
    return w-0.00001/numExamples * np.sum(gradients)
    

print "\nMulticlass Classifer trained on iris dataset\n"

start_time = time.time()
x, y, numLabels = parseData()
np.random.seed(0)
w = np.random.rand(x.shape[1], numLabels)

for i in range(4000):
    w = updateWeights(w, x, y)
    if i%400 == 0:
        print "Step ", i, "\t", cost(w, x, y)
print "\nTraining Time: ", (time.time() - start_time)*1000, "ms\n"