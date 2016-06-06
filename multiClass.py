
import numpy as np

def parseData():
    csv = np.loadtxt ('iris.csv', delimiter=',', dtype = str)
    y = csv[:,-1]
    #print y[0], x[0]
    allLabelNames = list(set(y.flat))
    #print allLabelNames
    y = np.array([float(allLabelNames.index(i)) for i in y])
    return csv[:,:-1].astype(np.float), y, len(allLabelNames)

def cost(w,x,y):
#    print predict(w,x).shape
#    print y.shape
    numExamples = x.shape[0]
    predictions = predict(w, x)
    positiveAccuracy = y*np.log10(predictions)
    negativeAccuracy = (1-y)*np.log10(1-predictions)
#    print positiveAccuracy.shape, negativeAccuracy.shape
    return -1.0/numExamples*np.sum(positiveAccuracy+negativeAccuracy)

def predict(w,x):
    predictions = 1.0/(1.0+np.exp(np.dot(x, -w)))
    predictions[predictions < 10**-10] = 10**-10
    predictions[predictions >= 1] = 1-10**-10
    return predictions

def updateWeights(w,x,y):
    numExamples = x.shape[0]
#    print predict(w,x).shape
    gradients = np.dot((predict(w, x)-y), x)
    return w-0.001/numExamples * np.sum(gradients)
    
x, y, numLabels = parseData()
#print y[0], x[0]

np.random.seed(0)
w = np.random.rand(x.shape[1], numLabels)
print x.shape
print y.shape
print w.shape
#print cost(w, x, y)
#for i in range(100):
#    w = updateWeights(w, x, y)
#    print cost(w, x, y)