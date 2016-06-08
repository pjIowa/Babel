
import numpy as np
import time

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
    x = csv[:,:-1].astype(np.float)
    return np.hstack((np.ones((x.shape[0], 1)), x)), y, numLabels

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
    return w-0.00001/numExamples * np.sum(gradients)
    

print "\nMulticlass Classifer trained on iris dataset\n"
start_time = time.time()
x, y, numLabels = parseData()
#print y[0], x[0]
np.random.seed(0)
w = np.random.rand(x.shape[1], numLabels)
#print x.shape
#print y.shape
#print w.shape
#print w
#print np.sum((y - predict(w,x))**2)
#print cost(w, x, y)
for i in range(4000):
    w = updateWeights(w, x, y)
    if i%400 == 0:
        print "Step ", i, "\t", cost(w, x, y)
print "\nTraining Time: ", (time.time() - start_time)*1000, "ms\n"
#print np.sum((y - predict(w,x))**2)