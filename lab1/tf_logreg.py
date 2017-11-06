import tensorflow as tf
import numpy as np

class TFLogreg:

  def __init__(self, D, C, param_delta=0.5, param_lambda=1):
    """Arguments:
       - D: dimensions of each datapoint
       - C: number of classes
       - param_delta: training step
    """

    self.X = tf.placeholder(tf.float32, [None, D])
    self.Yoh = tf.placeholder(tf.float32, [None, C])

    self.w = tf.Variable(tf.zeros([D, C]))
    self.b = tf.Variable(tf.zeros([C]))

    self.logits = tf.matmul(self.X, self.w) + self.b
    self.Yp = tf.nn.softmax(self.logits)

    # self.loss = tf.reduce_mean(-tf.reduce_sum(self.Yoh * tf.log(self.Yp), reduction_indices=1)) + param_lambda * tf.nn.l2_loss(self.w)
    self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Yoh)) + param_lambda * tf.nn.l2_loss(self.w)

    self.trainer = tf.train.GradientDescentOptimizer(param_delta)
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
        _, loss_val = self.sess.run([self.train_op, self.loss], feed_dict={self.X: X, self.Yoh: Yoh_})
        print("loss={}: i={}".format(loss_val, i))

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
    probs = self.eval(X)
    return np.argmax(probs, axis=1)

