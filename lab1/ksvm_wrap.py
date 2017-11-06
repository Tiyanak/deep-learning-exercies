import sklearn.svm

class KSVMWrap:

    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto', kernel='rbf', decision_function_shape='ovo'):

        self.svm = sklearn.svm.SVC(gamma=param_svm_gamma, C=param_svm_c, kernel=kernel, decision_function_shape=decision_function_shape)

        self.svm.fit(X, Y_)

    def predict(self, X):

        return self.svm.predict(X)

    def get_scores(self, X, Y):

        return self.svm.score(X, Y)

    def support(self):

        return self.svm.support_

    def get_wights(self):

        return (self.svm.coef_, self.svm.intercept_)

