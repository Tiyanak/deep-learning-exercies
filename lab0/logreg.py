import numpy as np

class LogReg:

    def __init__(self, param_niter=1000, param_delta=0.5):
        self.param_niter = param_niter
        self.param_delta = param_delta

    def sigmoid_array(self, x):
        return 1 / (1 + np.exp(-x))

    def logreg_train(self, X, Y_):

        w = np.random.randn(len(X[0]))
        b = 0

        for i in range(0, self.param_niter):

            scores = np.dot(X, w) + b
            # expscores =

            # sumexp =

            probs = self.sigmoid_array(scores)
            logprobs = np.log(probs)

            loss = np.sum(-logprobs)

            # dijagnostički ispis
            if i % 10 == 0:
                print("iteration {}: loss {}".format(i, loss))

            # dL_ds =

            grad_W = (((probs - Y_) * X.T).T).sum(axis=0);
            grad_b = np.sum(probs - Y_)

            w += -self.param_delta * grad_W
            b += -self.param_delta * grad_b

    def logreg_classify(self, X, w, b):

        scores = np.dot(X, w) + b
        probs = self.sigmoid_array(scores)

        return probs


