import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from lab4 import utils
from lab4.RBM import RBM

class DBN():

    def __init__(self, Nh, h1_shape):

        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.train_x, self.train_y = self.mnist.train.images, self.mnist.train.labels
        self.test_x, self.test_y = self.mnist.test.images, self.mnist.test.labels

        self.Nh2 = Nh  # Broj elemenata drugog skrivenog sloja
        self.h2_shape = h1_shape

        self.Nu = 5000  # broj uzoraka za vizualizaciju rekonstrukcije
        self.batch_size = 100
        self.epochs = 100
        self.n_samples = self.mnist.train.num_examples
        self.total_batch = int(self.n_samples / self.batch_size) * self.epochs

    def init_model(self):

        self.rbm1 = RBM()
        self.rbm1.init_model()

        self.rbm2 = RBM()
        self.rbm2.set_init_weghts(self.rbm1.w, self.rbm1.vb, self.Nh2)
        self.rbm2.init_model()

        self.initializer = tf.global_variables_initializer()

    def init_session(self):

        self.sess = tf.Session()
        self.sess.run(self.initializer)

    def train(self):

        for i in range(self.total_batch):
            batch, label = self.mnist.train.next_batch(self.batch_size)
            err, _ = self.sess.run([self.rbm1.err_sum, self.rbm1.out], feed_dict={self.rbm1.X: batch})

            if i % (int(self.total_batch / 10)) == 0:
                print(i, err)

        for i in range(self.total_batch):
            batch, label = self.mnist.train.next_batch(self.batch_size)
            err, _ = self.sess.run([self.rbm2.err_sum, self.rbm2.out], feed_dict={self.rbm2.X: batch})

            if i % (int(self.total_batch / 10)) == 0:
                print(i, err)

        w2s = self.rbm2.w.eval(session=self.sess)
        vb2s = self.rbm2.vb.eval(session=self.sess)
        vr2, h2s = self.sess.run([self.rbm2.v_prob, self.rbm2.h], feed_dict={self.rbm2.X: self.test_x[0: self.Nu, :]})

        return (w2s, vb2s, vr2, h2s)

    def visualize(self, w, v, h):

        # vizualizacija težina
        utils.draw_weights(w, self.rbm1.v_shape, self.Nh2, self.h2_shape, name="dbn_weights.png", interpolation="nearest")

        # vizualizacija rekonstrukcije i stanja
        utils.draw_reconstructions(self.test_x, v, h, self.rbm1.v_shape, self.h2_shape, 20, name="dbn_reconstructions.png")