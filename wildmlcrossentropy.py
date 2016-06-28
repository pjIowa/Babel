import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model

class WML:
    
    def __init__(self):
        self.loadData()
        self.loadWeights()
        np.set_printoptions(suppress=True)
        
    def loadData(self):
        np.random.seed(0)
        self.num_examples = 200
        self.X, self.y = sklearn.datasets.make_moons(self.num_examples, noise=0.20)
        # add bias term
        self.X = np.hstack((np.ones((self.num_examples, 1)), self.X))
    
    def loadWeights(self):
        nn_input_dim = 3 # input layer dimensionality
        nn_output_dim = 2 # output layer dimensionality
        nn_hidden_dim = 100 # hidden layer dimensionality

        self.epsilon = 0.01 # learning rate for gradient descent
        self.reg_lambda = 0.01 # regularization strength

        self.w1 = np.random.randn(nn_input_dim, nn_hidden_dim)
        self.w2 = np.random.randn(nn_hidden_dim, nn_output_dim)

    def calculate_loss(self):
        # Forward propagation to calculate our predictions
        z1 = self.X.dot(self.w1)
        a1 = np.tanh(z1)
        z2 = a1.dot(self.w2)
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # Calculating the loss
        correct_logprobs = -np.log(probs[range(self.num_examples), self.y])
        data_loss = np.sum(correct_logprobs)
        
        # Add regulatization term to loss (optional)
        data_loss += self.reg_lambda/2 * (np.sum(np.square(self.w1)) + np.sum(np.square(self.w2)))
        return 1./self.num_examples * data_loss
    
    def predict(self):
        # Forward propagation
        z1 = self.X.dot(self.w1)
        a1 = np.tanh(z1)
        z2 = a1.dot(self.w2)
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        print np.argmax(probs, axis=1)
        print self.y
        return np.argmax(probs, axis=1)
    
    def updateWeights(self):
        # Forward propagation
        z1 = self.X.dot(self.w1)
        a1 = np.tanh(z1)
        z2 = a1.dot(self.w2)
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # Backpropagation
        probs[range(self.num_examples), self.y] -= 1
        dW2 = (a1.T).dot(probs)
        delta2 = probs.dot(self.w2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(self.X.T, delta2)
 
        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += self.reg_lambda * self.w2
        dW1 += self.reg_lambda * self.w1
 
        # Gradient descent parameter update
        self.w1 += -self.epsilon * dW1
        self.w2 += -self.epsilon * dW2
    

w = WML()
w.predict()
for i in range(20000):
    if i%1000 == 0:
        print w.calculate_loss()
    w.updateWeights()
w.predict()