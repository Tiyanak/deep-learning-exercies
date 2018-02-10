import tensorflow as tf
import matplotlib.pyplot as plt
from lab4 import utils
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class DBN_Fine_Tune():

    def __init__(self):

        self.init_params()
        self.init_data()

    def init_params(self):

        self.Nv = 784  # broj elemenata vidljivog sloja
        self.v_shape = (28, 28)  # velicina slike
        self.Nh1 = 100  # broj elemenata 1 skrivenog sloja
        self.h1_shape = (10, 10)  # 2D prikaz broja stanja
        self.Nh2 = 100  # broj elemenata 2 skrivenog sloja
        self.h2_shape = (10, 10)  # 2D prikaz broja stanja

        self.Nu = 5000  # broj uzoraka za vizualizaciju rekonstrukcije
        self.gibbs_sampling_steps = 2
        self.alpha = 1
        self.beta = 0.01
        self.batch_size = 100
        self.epochs = 100

    def init_data(self):

        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.train_x, self.train_y = self.mnist.train.images, self.mnist.train.labels
        self.test_x, self.test_y = self.mnist.test.images, self.mnist.test.labels

        self.n_samples = self.mnist.train.num_examples
        self.total_batch = int(self.n_samples / self.batch_size) * self.epochs

    def init_model(self, w1, w2, v_bias, h1_bias_up, h1_bias_down, h2_bias):

        self.X3 = tf.placeholder("float", [None, self.Nv])
        self.R1 = tf.Variable(w1)
        self.W1 = tf.Variable(tf.transpose(w1))
        self.W2 = tf.Variable(w2)
        self.h1_bias_up = tf.Variable(h1_bias_up)
        self.h1_bias_down = tf.Variable(h1_bias_down)
        self.v_bias = tf.Variable(v_bias)
        self.h2_bias = tf.Variable(h2_bias)

        # wake pass
        self.h1_up_prob = tf.nn.sigmoid(tf.matmul(self.X3, self.R1) + self.h1_bias_up) # dopuniti
        self.h1_up = utils.sample_prob(self.h1_up_prob) # s^{(n)} u pripremi
        self.v_up_down_prob = tf.nn.sigmoid(tf.matmul(self.h1_up, self.W1) + self.v_bias)
        self.v_up_down = utils.sample_prob(self.v_up_down_prob) # s^{(n-1)\mathit{novo}} u tekstu pripreme

        # top RBM Gibs passes
        self.h2_up_prob = tf.nn.sigmoid(tf.matmul(self.h1_up, self.W2) + self.h2_bias)
        self.h2_up = utils.sample_prob(self.h2_up_prob)
        self.h2_down = self.h2_up

        for step in range(self.gibbs_sampling_steps):
            self.h1_down_prob = tf.nn.sigmoid(tf.matmul(self.h2_down, tf.transpose(self.W2)) + self.h1_bias_down)  # dopuniti
            self.h1_down = utils.sample_prob(self.h1_down_prob)  # dopuniti
            self.h2_down_prob = tf.nn.sigmoid(tf.matmul(self.h1_down, self.W2) + self.h2_bias)  # dopuniti
            self.h2_down = utils.sample_prob(self.h2_down_prob)  # dopuniti

        # sleep pass
        self.v_down_prob = tf.nn.sigmoid(tf.matmul(self.h1_up, self.W1) + self.v_bias)
        self.v_down = utils.sample_prob(self.v_down_prob) # s^{(n-1)} u pripremi
        self.h1_down_up_prob = tf.nn.sigmoid(tf.matmul(self.v_down, self.R1) + self.h1_bias_up)
        self.h1_down_up = utils.sample_prob(self.h1_down_up_prob) # s^{(n)\mathit{novo}} u pripremi

        # generative weights update during wake pass
        self.update_w1_down = tf.assign_add(self.W1,
                                       self.beta * tf.matmul(tf.transpose(self.h1_up),
                                       self.X3 - self.v_up_down_prob) / tf.to_float(tf.shape(self.X3)[0]))
        self.update_vb1_down = tf.assign_add(self.v_bias, self.beta * tf.reduce_mean(self.X3 - self.v_up_down_prob, 0))

        # top RBM update
        self.w2_positive_grad = tf.matmul(tf.transpose(self.h1_up), self.h2_up) # dopuniti
        self.w2_negative_grad = tf.matmul(tf.transpose(self.h1_down), self.h2_down) # dopuniti

        self.dw = (self.w2_positive_grad - self.w2_negative_grad) / tf.to_float(tf.shape(self.h1_up)[0])

        self.update_w2 = tf.assign_add(self.W2, self.beta * self.dw)
        self.update_hb1_down = tf.assign_add(self.h1_bias_down, self.beta * tf.reduce_mean(self.h1_up - self.h1_down, 0))
        self.update_hb2 = tf.assign_add(self.h2_bias, self.beta * tf.reduce_mean(self.h2_up - self.h2_down, 0))

        # recognition weights update during sleep pass
        self.update_r1_up = tf.assign_add(self.R1,
                                     self.beta * tf.matmul(tf.transpose(self.v_down_prob),
                                     self.h1_down - self.h1_down_up) / tf.to_float(tf.shape(self.X3)[0]))
        self.update_hb1_up = tf.assign_add(self.h1_bias_up, self.beta * tf.reduce_mean(self.h1_down - self.h1_down_up, 0))

        self.out = (self.update_w1_down, self.update_vb1_down, self.update_w2,
                    self.update_hb1_down, self.update_hb2, self.update_r1_up, self.update_hb1_up)

        self.err = self.X3 - self.v_down_prob
        self.err_sum = tf.reduce_mean(self.err * self.err)

        self.initialize = tf.global_variables_initializer()

    def init_session(self):

        self.sess = tf.Session()
        self.sess.run(self.initialize)

    def train(self):

        for i in range(self.total_batch):
            batch, label = self.mnist.train.next_batch(self.batch_size)
            err, _ = self.sess.run([self.err_sum, self.out], feed_dict={self.X3: batch})

            if i % (int(self.total_batch / 10)) == 0:
                print(i, err)

        R1, W1, W2, v_bias, h1_bias_up, h1_bias_down, h2_bias = self.sess.run(
            [self.R1, self.W1, self.W2, self.v_bias, self.h1_bias_up, self.h1_bias_down, self.h2_bias],
            feed_dict={self.X3: batch}
        )

        v_out_prob, h_top, h_top_prob = self.sess.run(
            [self.v_down_prob, self.h2_down, self.h2_down_prob],
            feed_dict={self.X3: self.test_x[0:self.Nu, :]})

        return R1, W1, W2, v_bias, h1_bias_up, h1_bias_down, h2_bias, v_out_prob, h_top, h_top_prob

    def visualize(self, R1, W1, W2, v_bias, h1_bias_up, h1_bias_down, h2_bias, v_out_prob1, v_out_prob2, v_out_prob3,
                  h_top, h_top_prob):

        # vizualizacija težina
        utils.draw_weights(R1, self.v_shape, self.Nh1, self.h1_shape, name="dbnFT_R1_weights.png")
        utils.draw_weights(W1.T, self.v_shape, self.Nh1, self.h1_shape, name="dbnFT_W1_weights.png")
        utils.draw_weights(W2, self.h1_shape, self.Nh2, self.h2_shape, name="dbnFT_W2_weights.png", interpolation="nearest")

        # vizualizacija rekonstrukcije i stanja
        utils.draw_reconstructions(self.test_x, v_out_prob3, h_top, self.v_shape, self.h2_shape, 20,
                                   name="dbnFT_reconstructions.png")

        self.reconstruct(v_out_prob1, v_out_prob2, v_out_prob3, h_top, name="dbnFT_outs_reconstructions.png")

        r_input = self.generate_patterns()

        out = self.sess.run((self.v_down), feed_dict={self.h1_up: r_input})

        # Emulacija dodatnih Gibbsovih uzorkovanja pomocu feed_dict
        for i in range(1000):
            out_prob, out, hout = self.sess.run((self.v_down_prob, self.v_down, self.h2_down), feed_dict={self.X3: out})

        utils.draw_generated(r_input, hout, out_prob, self.v_shape, self.h2_shape, 50, name="dbnFT_generated.png")

    def generate_patterns(self):

        # Generiranje uzoraka iz slucajnih vektora
        r_input = np.random.rand(100, self.Nh2)
        r_input[r_input > 0.9] = 1  # postotak aktivnih - slobodno varirajte
        r_input[r_input < 1] = 0
        r_input = r_input * 20  # pojacanje za slucaj ako je mali postotak aktivnih

        s = 10
        i = 0
        r_input[i, :] = 0
        r_input[i, i] = s
        i += 1
        r_input[i, :] = 0
        r_input[i, i] = s
        i += 1
        r_input[i, :] = 0
        r_input[i, i] = s
        i += 1
        r_input[i, :] = 0
        r_input[i, i] = s
        i += 1
        r_input[i, :] = 0
        r_input[i, i] = s
        i += 1
        r_input[i, :] = 0
        r_input[i, i] = s
        i += 1
        r_input[i, :] = 0
        r_input[i, i] = s

        return r_input

    def reconstruct(self, vr, vr2, vr3, h2_downs, name="dbnFT_outs_reconstructions.png"):

        # vizualizacija rekonstrukcije i stanja
        Npics = 5
        plt.figure(figsize=(8, 12 * 4))
        for i in range(20):
            plt.subplot(20, Npics, Npics * i + 1)
            plt.imshow(self.test_x[i].reshape(self.v_shape), vmin=0, vmax=1)
            plt.title("Test input")
            plt.subplot(20, Npics, Npics * i + 2)
            plt.imshow(vr[i][0:784].reshape(self.v_shape), vmin=0, vmax=1)
            plt.title("Reconstruction 1")
            plt.subplot(20, Npics, Npics * i + 3)
            plt.imshow(vr2[i][0:784].reshape(self.v_shape), vmin=0, vmax=1)
            plt.title("Reconstruction 2")
            plt.subplot(20, Npics, Npics * i + 4)
            plt.imshow(vr3[i][0:784].reshape(self.v_shape), vmin=0, vmax=1)
            plt.title("Reconstruction 3")
            plt.subplot(20, Npics, Npics * i + 5)
            plt.imshow(h2_downs[i][0:self.Nh2].reshape(self.h2_shape), vmin=0, vmax=1, interpolation="nearest")
            plt.title("Top states 3")
        plt.tight_layout()
        plt.savefig(name)


