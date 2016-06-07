
import numpy as np

def parseData():
    csv = np.loadtxt ('iris.csv', delimiter=',', dtype = str)
    y = csv[:,-1]
    #print y[0], x[0]
    allLabelNames = sorted(set(y.flat))
    numLabels = len(allLabelNames)
    #print allLabelNames
#    print y[0], y[51], y[101]
    y = np.array([[1.0*(elem==j) for j in allLabelNames] for elem in y])
#    print y[0], y[51], y[101]
    return csv[:,:-1].astype(np.float), y, numLabels

def cost(w,x,y):
#    print predict(w,x).shape
#    print y.shape
    numExamples = x.shape[0]
#    print w.shape[1]
    predictions = predict(w, x)
#    print predictions.shape, y.shape
#    print predictions[0], y[0]
    positiveAccuracy = y*np.log10(predictions)
    negativeAccuracy = (1-y)*np.log10(1-predictions)
#    print positiveAccuracy.shape, negativeAccuracy.shape
    return -1.0/numExamples*np.sum(positiveAccuracy+negativeAccuracy)

def predict(w,x):
    predictions = 1.0/(1.0+np.exp(np.dot(x, -w)))
    predictions[predictions < 10**-10] = 10**-10
    predictions[predictions >= 1] = 1-10**-10
#    print predictions.shape
    return predictions

def updateWeights(w,x,y):
    numExamples = x.shape[0]
#    print predict(w,x).shape
    gradients = np.dot((predict(w, x)-y).T, x).T
#    print gradients.shape
    return w-0.001/numExamples * np.sum(gradients)
    
x, y, numLabels = parseData()
#print y[0], x[0]

np.random.seed(0)
w = np.random.rand(x.shape[1], numLabels)
#print x.shape
#print y.shape
#print w.shape
#print w
print cost(w, x, y)
for i in range(100):
    w = updateWeights(w, x, y)
#    print w
    print cost(w, x, y)

#data = np.array([0,1,2,0,1,2])
#
#print data
#print range(numLabels)
#print np.array([[1.0*elem==j for j in range(numLabels)] for elem in data], dtype=float)
#print [i in numLabels for i in data]