from data import *
from sklearn.linear_model import LogisticRegression
from lab1.fcann2 import Fcann2
from lab1.tf_linreg import Tf_linreg
import tensorflow as tf
from lab1.tf_logreg import TFLogreg
from lab1.tf_deep import TfDeep
from lab1.ksvm_wrap import KSVMWrap
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split

def zadatak1():
    X, Y = data.sample_gmm(6, 2, 10)
    X2, Y2 = data.sample_gmm(6, 2, 10)

    logreg = LogisticRegression()
    logreg.fit_transform(X, Y)
    Yp = logreg.predict(X2)

    data.plot_data(X2, Y2, Yp, logreg)

    plt.show()

def zadatak2():
    X, Y = data.sample_gmm(6, 2, 10)
    X2, Y2 = data.sample_gmm(6, 2, 10)

    fcann2 = Fcann2(param_lambda=1)
    fcann2.fcann2_train(X, Y)
    Yp = fcann2.predict(X)

    acc, pr, m = data.eval_perf_multi(Y, Yp)
    print("acc={}, recall={}, precision={}".format(acc, pr[0][0], pr[0][1]))

    data.plot_data(X, Y, Yp, fcann2)

    plt.show()

def zadatak3():

    tf_linreg = Tf_linreg()
    # a)
    w, b = tf_linreg.train([1, 2], [3, 5])

    print("w {}: b {}".format(w, b))

    X = np.linspace(0, 10, 40)
    Y = X * w + b

    plt.plot(X, Y, '--', linewidth=2)
    plt.scatter([1, 2], [3, 5])
    plt.show()

    # b) nebi znao pronaci pravac sa puno tocaka gaussove mjesavine
    # X, Y = (np.random.randn(8), np.random.randn(8))
    # w1, b1 = tf_linreg.train(X, Y)
    #
    # print("w {}: b {}".format(w1, b1))
    #
    # Xs = np.linspace(0, 10, 8)
    # Ys = Xs * w1 + b1
    #
    # plt.plot(Xs, Ys, '--', linewidth=2)
    # plt.scatter(X, Y)
    # plt.show()

def zadatak4():

    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)
    tf.set_random_seed(100)

    C = 2
    X, Y = data.sample_gmm(6, C, 10)
    Yoh = data.class_to_onehot(Y)

    # izgradi graf:
    tflr1 = TFLogreg(X.shape[1], Yoh.shape[1], param_lambda=1)

    # nauči parametre:
    tflr1.train(X, Yoh, 10000)

    # dohvati vjerojatnosti na skupu za učenje
    probs = tflr1.eval(X)
    predicts = np.argmax(probs, axis=1)

    # ispiši performansu (preciznost i odziv)
    acc, pr, m = data.eval_perf_multi(Y, predicts)
    print("accuracy={}, precision={}, recall={}".format(acc, pr[0][0], pr[0][1]))

    data.plot_data(X, Y, predicts, tflr1)

    plt.show()

def zadatak5():

    np.random.seed(100)
    tf.set_random_seed(100)

    C = 3
    X, Y = data.sample_gmm(6, C, 10)
    Yoh = data.class_to_onehot(Y)

    tflr2 = TfDeep([len(X[0]), C], 0.1, 1)

    tflr2.train(X, Yoh, 1000)

    probs = tflr2.eval(X)
    predicts = tflr2.predict(X)

    acc, pr, m = data.eval_perf_multi(Y, predicts)
    print("accuracy={}, precision={}, recall={}".format(acc, pr[0][0], pr[0][1]))

    data.plot_data(X, Y, predicts, tflr2)

    tflr2.count_params()

    plt.show()

def zadatak6():
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)
    tf.set_random_seed(100)

    C = 3
    X, Y = data.sample_gmm(6, C, 10)

    ksvm = KSVMWrap(X, Y)

    predicts = ksvm.predict(X)
    score = ksvm.get_scores(X, Y)

    acc, pr, m = data.eval_perf_multi(Y, predicts)
    print("accuracy={}, precision={}, recall={}, score={}".format(acc, pr[0][0], pr[0][1], score))

    data.plot_data(X, Y, predicts, ksvm, ksvm.support())

    plt.show()


def zadatak7():

    tf.app.flags.DEFINE_string('data_dir',
                               '/tmp/dataset/', 'Directory for storing dataset')
    mnist = input_data.read_data_sets(
        tf.app.flags.FLAGS.data_dir, one_hot=True)

    images = mnist.train.images
    labels = mnist.train.labels

    np.random.seed(100)
    tf.set_random_seed(100)

    C = len(labels[0])

    # tflr2 = TfDeep([len(images[0]), C], 0.1, 1)
    #
    # tflr2.train(images, labels, 100)
    #
    # probs = tflr2.eval(images)
    # oneImage = np.array([images[0]])
    # predicts = tflr2.predict(mnist.train.images)
    #
    # labelsTest = np.argmax(mnist.train.labels, axis=1)
    #
    # acc, pr, m = dataset.eval_perf_multi(labelsTest, predicts)
    # print("accuracy={}, precision={}, recall={}".format(acc, pr[0][0], pr[0][1]))
    #
    # print("pred = {} : label = {}".format(predicts, labelsTest))
    #
    # w, b = tflr2.get_weights();
    # plt.figure(1)
    # total_w = []
    # dataset.plot_weights(w)

    # 1/5 za validaciju
    # X_train, X_val, Y_train, Y_val = train_test_split(mnist.train.images, mnist.train.labels, train_size=4.0/5, random_state=100)
    #
    # tflr2 = TfDeep([len(images[0]), C], 0.1, 1)
    # tflr2.train(images, labels, 100)
    # labels_val = np.argmax(Y_val, axis=1)
    # predicts_val = tflr2.predict(X_val)
    #
    # acc_val, pr_val, m_val = dataset.eval_perf_multi(labels_val, predicts_val)
    # print("accuracy={}, precision={}, recall={}".format(acc_val, pr_val[0][0], pr_val[0][1]))
    #
    # predicts_test = tflr2.predict(mnist.test.images)
    # labels_test = np.argmax(mnist.test.labels, axis=1)
    #
    # acc_test, pr_test, m_test = dataset.eval_perf_multi(labels_test, predicts_test)
    # print("accuracy={}, precision={}, recall={}".format(acc_test, pr_test[0][0], pr_test[0][1]))
    #
    # w, b = tflr2.get_weights();
    # plt.figure(2)
    # total_w = []
    # dataset.plot_weights(w)

    # batch eval

    tflr2 = TfDeep([len(images[0]), C], 0.1, 2)
    tflr2.train_mb(images, labels, 100, 3)

    predicts = tflr2.predict(mnist.test.images)
    labelsTest = np.argmax(mnist.test.labels, axis=1)

    acc, pr, m = data.eval_perf_multi(labelsTest, predicts)
    print("accuracy={}, precision={}, recall={}".format(acc, pr[0][0], pr[0][1]))

    w, b = tflr2.get_weights();

    plt.figure(3)
    data.plot_weights(w)

    # SVM usporedba

    # lin
    # ksvm1 = KSVMWrap(mnist.train.images, np.argmax(mnist.train.labels, axis=1))
    # predicts = ksvm1.predict(mnist.test.images)
    # acc, pr, m = dataset.eval_perf_multi(np.argmax(mnist.test.labels, axis=1), predicts)
    # print("accuracy={}, precision={}, recall={}".format(acc, pr[0][0], pr[0][1]))
    #
    # dataset.plot_data(mnist.train.images, np.argmax(mnist.train.labels, axis=1), predicts, ksvm1)
    #
    # ksvm2 = KSVMWrap(mnist.train.images, np.argmax(mnist.train.labels, axis=1), kernel='linear')
    # predicts = ksvm2.predict(mnist.test.images)
    # acc, pr, m = dataset.eval_perf_multi(np.argmax(mnist.test.labels, axis=1), predicts)
    # print("accuracy={}, precision={}, recall={}".format(acc, pr[0][0], pr[0][1]))
    #
    # plt.figure(4)
    # dataset.plot_weights(ksvm2.get_wights()[0])

    plt.show()

if __name__ == "__main__":

    data = Random2DGaussian()

    print("####### zad 1 #######")
    # plt.figure(1)
    # zadatak1()

    print("####### zad 2 #######")
    # plt.figure(2)
    # zadatak2()

    print("####### zad 3 #######")
    # plt.figure(3)
    # zadatak3()

    print("####### zad 4 #######")
    # plt.figure(4)
    # zadatak4()

    print("####### zad 5 #######")
    # plt.figure(5)
    # zadatak5()

    print("####### zad 6 #######")
    # plt.figure(6)
    zadatak6()

    print("####### zad 7 #######")
    # plt.figure(7)
    # zadatak7()