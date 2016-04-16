import numpy as np

class RNN:
     
    def __init__(self, iDim, oDim, hDim=2):
        self.iDim = iDim
        self.hDim = hDim
        self.oDim = oDim
        self.U = np.random.uniform(-np.sqrt(1./iDim), np.sqrt(1./iDim), (hDim, iDim))
        self.V = np.random.uniform(-np.sqrt(1./hDim), np.sqrt(1./hDim), (oDim, hDim))
        self.W = np.random.uniform(-np.sqrt(1./hDim), np.sqrt(1./hDim), (hDim, hDim))
    
    def forwardPropagation(self, x):
        T = len(x)
    
        s = np.zeros((T + 1, self.hDim))
        s[-1] = np.zeros(self.hDim)
    
        o = np.zeros((T, self.iDim))
        
        for t in np.arange(T):
            s[t] = np.tanh(self.U[:,t].dot(x) + self.W.dot(s[t-1]))
            o[t] = self.V.dot(s[t])
        return [o, s]
    
    def predict(self, x):
        o, s = self.forwardPropagation(x)
        return np.sum(o,axis=0)

    
def sumXOR(x):
    ret = []
    for i in range(1, len(x)):
        ret.append(np.logical_xor(x[i]==1, x[i-1]==1))
    return sum(map(int,ret))

sequenceLength = 10
m = 20
outputLength = 1

xTrain = np.around(np.random.rand(m, sequenceLength))
yTrain = np.apply_along_axis( sumXOR, axis=1, arr=xTrain )
#print x
#print y
# x_t, input at each timestep: [10]
# y_t, output at each timestep: [1]
# s_t, context at each timestep: [1,2]
# u, input weights: [2,10]
# v, output weights: [1,2]
# w, context weights: [2,2]

# s_t = tanh(u*x_t + w*s_t-1)
# u*x_t = 2x10x10x1 = 2x1
# w*s_t-1 = 2x2x2x1

# o_t = softmax(v*s_t)
# v*s_t = 1x2x2x1 = 1

np.random.seed(10)
model = RNN(sequenceLength,outputLength)
#o, s = model.forwardPropagation(xTrain[0])
predictions = model.predict(xTrain[10])
print xTrain[0].shape
print predictions.shape
print predictions

#print o.shape
#print xTrain.shape
#print xTrain[0].shape
#print xTrain[0]
#print o




