import tensorflow as tf
import numpy as np
from sklearn.model_selection import StratifiedKFold

class TfDeep:
    def __init__(self, layers_size=[2, 2], param_delta=0.0001, param_lambda=0.5):
        """Arguments:
           - D: dimensions of each datapoint 
           - C: number of classes
           - param_delta: training step
        """
        self.X = tf.placeholder(tf.float32, [None, layers_size[0]], 'X')
        self.Yoh = tf.placeholder(tf.float32, [None, layers_size[len(layers_size)-1]], 'Yoh')

        losses = []
        h = self.X

        self.total_w = []

        for layer_index in range(0, len(layers_size)-1):

            l_this = layers_size[layer_index]
            l_next = layers_size[layer_index+1]

            self.w = tf.Variable(tf.random_normal([l_this, l_next], 0, 1.0), 'W')
            self.b = tf.Variable(tf.constant(0, tf.float32, [l_next]), 'b')

            self.total_w.append(self.w)

            losses.append(tf.nn.l2_loss(self.w))
            h = tf.matmul(h, self.w) + self.b

            if layer_index + 1 < len(layers_size):
                h = tf.nn.sigmoid(h)

        self.logits = h
        self.Yp = tf.nn.softmax(self.logits)

        # self.loss = tf.reduce_mean(-tf.reduce_sum(self.Yoh * tf.log(self.Yp), reduction_indices=1)) + param_lambda * tf.add_n(losses)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                           labels=self.Yoh)) + param_lambda * tf.add_n(losses)

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(param_delta, global_step,
                                                   1, 1-param_delta, staircase=True)

        self.trainer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = self.trainer.minimize(self.loss)

        self.sess = tf.Session()


    def train(self, X, Yoh_, param_niter):
        """Arguments:
           - X: actual datapoints [NxD]
           - Yoh_: one-hot encoded labels [NxC]
           - param_niter: number of iterations
        """

        self.sess.run(tf.global_variables_initializer())

        for i in range(param_niter):
            self.sess.run(self.train_op, feed_dict={self.X: X, self.Yoh: Yoh_})

    def eval(self, X):
        """Arguments:
           - X: actual datapoints [NxD]
           Returns: predicted class probabilites [NxC]
        """
        return self.sess.run(self.Yp, feed_dict={self.X: X})

    def predict(self, X):
        """Arguments:
               - X: actual datapoints [NxD]
               Returns: predicted class [Nx1]
            """
        return np.argmax(self.eval(X), axis=1)

    def count_params(self):
        for var in tf.trainable_variables():
            print("var = {}".format(var.name))

    def get_weights(self):

        return self.sess.run([self.total_w, self.b])

    def train_mb(self, X, Yoh_, param_niter=1000, n_groups=1):

        self.sess.run(tf.global_variables_initializer())

        for i in range(param_niter):

            group_kfold = StratifiedKFold(n_groups, shuffle=True, random_state=100)
            for train_index, test_index in group_kfold.split(X, np.argmax(Yoh_, axis=1)):

                X_train = np.vstack((X[train_index], X[test_index]))
                Y_train = np.vstack((Yoh_[train_index], Yoh_[test_index]))

                self.sess.run(self.train_op, feed_dict={self.X: X_train, self.Yoh: Y_train})
