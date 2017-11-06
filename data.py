import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import confusion_matrix

class Random2DGaussian:

    def __init__(self, minx=0, maxx=10, miny=0, maxy=10):
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy

        self.mi = np.array([np.random.uniform(minx, maxx), np.random.uniform(miny, maxy)])
        self.D = np.diag([(np.random.random_sample() * (maxx - minx) / 5) ** 2,
                          (np.random.random_sample() * (maxy - miny) / 5) ** 2])

        angle = np.random.uniform(0, 2 * np.pi)
        self.R = np.matrix([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

        self.sigma = np.dot(np.dot(self.R.T, self.D), self.R)

    def get_sample(self, n):
        return np.random.multivariate_normal(self.mi, self.sigma, n)

    def sample_gauss_2d(self, C, N):
        X = []
        Y = []
        for i in range(C):
            G = Random2DGaussian()
            X.append(G.get_sample(N))
            Y.append(np.vstack([i] * N))
        X = np.vstack(X)
        Y = np.vstack(Y)
        return X, Y

    def sample_gmm_2d(self, K, C, N):
        X, Y = [], []
        for i in range(K):
            G = Random2DGaussian()
            X.append(G.get_sample(N))
            Y.append(np.full((N, 1), np.random.randint(C)))
        X = np.vstack(X)
        Y = np.vstack(Y)
        return X, np.array(Y)

    def eval_perf_binary(self, Y, Y_):

        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        for i in range(0, len(Y)):
            if Y[i] == 1:
                if Y_[i] == 1:
                    true_positives += 1
                else:
                    false_negatives += 1
            else:
                if Y_[i] == 1:
                    false_positives += 1
                else:
                    true_negatives += 1

        accuracy = (float)(true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)
        precision = (float)(true_positives) / (true_positives + false_positives)
        recall = (float)(true_positives) / (true_positives + false_negatives)

        return (accuracy, precision, recall)

    def eval_perf_multi(self, Y, Y_):
        pr = []
        n = max(Y_) + 1
        # M = np.bincount(n * Y_ + Y, minlength=n * n).reshape(n, n)
        M = confusion_matrix(Y, Y_)
        for i in range(n):
            tp_i = M[i, i]
            fn_i = np.sum(M[i, :]) - tp_i
            fp_i = np.sum(M[:, i]) - tp_i
            tn_i = np.sum(M) - fp_i - fn_i - tp_i
            recall_i = tp_i / (tp_i + fn_i)
            precision_i = tp_i / (tp_i + fp_i)
            pr.append((recall_i, precision_i))

        accuracy = np.trace(M) / np.sum(M)

        return accuracy, pr, M

    def sample_gmm(self, ncomponents, nclasses, nsamples):
        # create the distributions and groundtruth labels
        Gs = []
        Ys = []
        for i in range(ncomponents):
            Gs.append(Random2DGaussian())
            Ys.append(np.random.randint(nclasses))

        # sample the dataset
        X = np.vstack([G.get_sample(nsamples) for G in Gs])
        Y_ = np.hstack([[Y] * nsamples for Y in Ys])

        return X, Y_

    def graph_data(self, X, Y_, Y):

        tocni = []
        netocni = []

        for i in range(0, len(Y)):

            if (Y_[i] == Y[i]):
                tocni.append(i)
            else:
                netocni.append(i)

        Xtocni = X[tocni]
        Xnetocni = X[netocni]

        plt.scatter(Xtocni[:, 0], Xtocni[:, 1], c=Y_[tocni], marker='o')
        plt.scatter(Xnetocni[:, 0], Xnetocni[:, 1], c=Y_[netocni], marker='s')

    def graph_data_2(self, X, Y_, Y, special=[]):

        palette = ([0.5, 0.5, 0.5], [1, 1, 1], [0.2, 0.2, 0.2])
        colors = np.tile([0.0, 0.0, 0.0], (Y_.shape[0], 1))
        for i in range(len(palette)):
            colors[Y_ == i] = palette[i]

        # sizes of the datapoint markers
        sizes = np.repeat(20, len(Y_))
        sizes[special] = 40

        # draw the correctly classified datapoints
        good = (Y_ == Y)
        plt.scatter(X[good, 0], X[good, 1], c=colors[good],
                    s=sizes[good], marker='o')

        # draw the incorrectly classified datapoints
        bad = (Y_ != Y)
        plt.scatter(X[bad, 0], X[bad, 1], c=colors[bad],
                    s=sizes[bad], marker='s')

    def class_to_onehot(self, Y):
        Yoh = np.zeros((len(Y), max(Y) + 1))
        Yoh[range(len(Y)), Y] = 1
        return Yoh



    def decisionFun(self, X):
        scores = X[:, 0] + X[:, 1] - (np.max(X) / 2)
        return scores

    def make_meshgrid(self, x, y, h=.02):

        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        return xx, yy

    def plot_contours(self, ax, clf, xx, yy, **params):

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out

    def plot_data(self, X, Y, Yp, model, special=[]):

        X0, X1 = X[:, 0], X[:, 1]

        xx, yy = self.make_meshgrid(X0, X1)

        self.plot_contours(plt, model, xx, yy, cmap=plt.get_cmap('seismic'), edgecolors='k')
        self.graph_data_2(X, Y, Yp, special)

    def plot_weights(self, w):

        total_w = []
        for w_sample in w:
            if len(total_w) == 0:
                total_w = w_sample
            else:
                total_w = np.matmul(total_w, w_sample)

        for i in range(0, 10):
            plt.subplot(2, 5, i + 1)
            weight = total_w[:, i]
            plt.title(i)
            plt.imshow(weight.reshape([28, 28]), cmap=plt.get_cmap('seismic'))
            frame1 = plt.gca()
            frame1.axes.get_xaxis().set_visible(False)
            frame1.axes.get_yaxis().set_visible(False)
