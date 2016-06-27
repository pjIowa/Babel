import numpy as np

def parseData():
    csv = np.genfromtxt ('diabetes.csv', delimiter=',')
    x = csv[:,:-1]
    return np.hstack((np.ones((x.shape[0], 1)), x)), csv[:,-1]

def cost(w,x,y):
    numExamples = x.shape[0]
    predictions = predict(w, x)
    positiveAccuracy = y*np.log(predictions)
    negativeAccuracy = (1-y)*np.log(1-predictions)
    return -1.0/numExamples*np.sum(positiveAccuracy+negativeAccuracy)

def predict(w,x):
    predictions = 1.0/(1.0+np.exp(np.dot(x, -w)))
    predictions[predictions < 10**-10] = 10**-10
    predictions[predictions >= 1] = 1-10**-10
    return predictions

def updateWeights(w,x,y):
    numExamples = x.shape[0]
    gradients = np.dot((predict(w, x)-y), x)
    return w-0.0001/numExamples * np.sum(gradients)
    
x, y = parseData()
np.random.seed(0)
w = np.random.rand(x.shape[1])

print cost(w, x, y)
for i in range(100):
    w = updateWeights(w, x, y)
    print cost(w, x, y)