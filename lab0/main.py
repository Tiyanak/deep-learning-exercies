from data import *
from lab0.binlogreg import *
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression

from data import *
from lab0.binlogreg import *

if __name__ == "__main__":
    G = Random2DGaussian()
    X = G.get_sample(100)

    Y_ = np.append([0] * 50, [1] * 50)

    L = LogReg();
    w, b = L.binlog_train(X, Y_)
    probs = L.binlogreg_classify(X, w, b)

    # print("w: {}\nb: {}".format(w, b))
    # print("probs: {}\n".format(probs))

    skL = LogisticRegression()
    skL.fit_transform(X, Y_)

    # print("w: {}\nb: {}".format(skL.coef_, skL.intercept_))
    # print("probs: {}\n".format(skL.predict_proba(X)))

    # X, Y_ = G.sample_gauss_2d(2, 100)
    # w, b = L.binlog_train(X, Y_)
    # probs = L.binlogreg_classify(X, w, b)
    # Y = [1 if i > 0.5 else 0 for i in probs]
    #
    # accuracy, recall, precision = G.eval_perf_binary(Y, Y_)
    # AP = G.eval_AP(Y_[probs.argsort()])
    # print(accuracy, recall, precision, AP)

    X, Y_ = G.sample_gauss_2d(2, 100)
    skL.fit_transform(X, Y_)
    Y = skL.predict(X)

    accuracy, recall, precision = G.eval_perf_binary(Y, Y_)
    # AP = G.eval_AP(Y_[probs.argsort()], Y[probs.argsort()])
    print(accuracy, recall, precision)

    G.graph_data(X, Y, Y_)
    plt.show()

