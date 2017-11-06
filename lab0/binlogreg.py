import numpy as np

class LogReg:

    def __init__(self, param_niter=1000, param_delta=0.5):
        self.param_niter = param_niter
        self.param_delta = param_delta

    def sigmoid_array(self, x):
        return 1 / (1 + np.exp(-x))

    def binlog_train(self, X, Y_):

        w = np.random.randn(len(X[0]))
        b = 0

        for i in range(0, self.param_niter):
            scores = np.dot(X, w) + b
            probs = self.sigmoid_array(scores)
            loss = np.sum(-np.log(probs))

            if i % 10 == 0:
                print("iteration {}: loss {}".format(i, loss))

            grad_w = (((probs - Y_) * X.T).T).sum(axis=0);
            grad_b = np.sum(probs - Y_)

            w += np.dot(-self.param_delta, grad_w)
            b += -self.param_delta * grad_b

        return (w, b)

    def binlogreg_classify(self, X, w, b):
        scores = np.dot(X, w) + b
        probs = self.sigmoid_array(scores)

        return probs

    def binlogreg_decfun(self, w, b):
        def classify(X):
            return self.binlogreg_classify(X, w, b)

        return classify