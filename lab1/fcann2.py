import numpy as np

class Fcann2:

    def __init__(self, param_niter=10000, param_delta=0.01, param_lambda=0.01):
        self.param_niter = param_niter
        self.param_delta = param_delta
        self.param_lambda = param_lambda
        self.W1 = []
        self.W2 = []
        self.b1 = []
        self.b2 = []

    def fcann2_train(self, X, Y):
        np.random.seed(100)

        self.W1 = np.random.randn(len(X[0]), 1)
        self.b1 = np.zeros((1, 1))
        self.W2 = np.random.randn(1, len(X[0]))
        self.b2 = np.zeros((1, len(X[0])))

        for i in range(0, self.param_niter):

            # forward
            s1 = X.dot(self.W1) + self.b1
            h1 = self.ReLU(s1)
            s2 = h1.dot(self.W2) + self.b2
            L = self.softmax(s2)

            # back
            dLdS = L
            dLdS[range(len(X)), Y] -= 1
            dW2 = (h1.T).dot(dLdS)
            db2 = np.sum(dLdS, axis=0)
            dLds1 = dLdS.dot(self.W2.T)
            dW1 = np.dot(X.T, dLds1)
            db1 = np.sum(dLds1, axis=0)

            dW2 += self.param_lambda * self.W2
            dW1 += self.param_lambda * self.W1

            self.W1 += -self.param_delta * dW1
            self.b1 += -self.param_delta * db1
            self.W2 += -self.param_delta * dW2
            self.b2 += -self.param_delta * db2

        return (self.W1, self.b1, self.W2, self.b2)

    def predict(self, X):
        s1 = X.dot(self.W1) + self.b1
        h1 = self.ReLU(s1)
        s2 = h1.dot(self.W2) + self.b2
        probs = self.softmax(s2)
        return np.argmax(probs, axis=1)


    def ReLU(self, vec):
        return np.maximum(0, vec)

    def softmax(self, z):
        ret = np.exp(z)
        ret /= np.sum(ret, axis=1, keepdims=True)

        return ret