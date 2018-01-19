import tensorflow as tf
import matplotlib.pyplot as plt
from lab4 import utils
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class DBN():

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
        self.gibbs_sampling_steps = 1
        self.alpha = 1
        self.batch_size = 100
        self.epochs = 100

    def init_data(self):

        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.train_x, self.train_y = self.mnist.train.images, self.mnist.train.labels
        self.test_x, self.test_y = self.mnist.test.images, self.mnist.test.labels

        self.n_samples = self.mnist.train.num_examples
        self.total_batch = int(self.n_samples / self.batch_size) * self.epochs

    def init_model(self, w1, vb, hb1):

        self.X2 = tf.placeholder("float", [None, self.Nv])
        self.w1 = tf.Variable(w1)
        self.v_bias = tf.Variable(vb)
        self.h1_bias = tf.Variable(hb1)
        self.w2 = utils.weights([self.Nh1, self.Nh2])
        self.h2_bias = utils.bias([self.Nh2])

        self.h1up_prob = tf.nn.sigmoid(tf.matmul(self.X2, self.w1) + self.h1_bias) # dopuniti
        self.h1up = utils.sample_prob(self.h1up_prob) # dopuniti
        self.h2up_prob = tf.nn.sigmoid(tf.matmul(self.h1up, self.w2) + self.h2_bias) # dopuniti
        self.h2up = utils.sample_prob(self.h2up_prob) # dopuniti
        self.h2down = self.h2up

        for step in range(self.gibbs_sampling_steps):
            self.h1down_prob = tf.nn.sigmoid(tf.matmul(self.h2down, tf.transpose(self.w2)) + self.h1_bias)  # dopuniti
            self.h1down = utils.sample_prob(self.h1down_prob)  # dopuniti
            self.h2down_prob = tf.nn.sigmoid(tf.matmul(self.h1down, self.w2) + self.h2_bias)  # dopuniti
            self.h2down = utils.sample_prob(self.h2down_prob)  # dopuniti

        w2_positive_grad = tf.matmul(tf.transpose(self.h1up), self.h2up) # dopuniti
        w2_negative_grad = tf.matmul(tf.transpose(self.h1down), self.h2down) # dopuniti

        dw2 = (w2_positive_grad - w2_negative_grad) / tf.to_float(tf.shape(self.h1up)[0])

        update_w2 = tf.assign_add(self.w2, self.alpha * dw2)
        update_hb1a = tf.assign_add(self.h1_bias, self.alpha * tf.reduce_mean(self.h1up - self.h1down, 0))
        update_hb2 = tf.assign_add(self.h2_bias, self.alpha * tf.reduce_mean(self.h2up - self.h2down, 0))

        self.out = (update_w2, update_hb1a, update_hb2)

        # rekonsturkcija ulaza na temelju krovnog skrivenog stanja h3
        self.h1_rec_prob = tf.nn.sigmoid(tf.matmul(self.h2down, tf.transpose(self.w2)) + self.h1_bias)
        self.h1_rec = utils.sample_prob(self.h1_rec_prob)
        self.v_out_prob = tf.nn.sigmoid(tf.matmul(self.h1_rec, tf.transpose(self.w1)) + self.v_bias) # dopuniti
        self.v_out = utils.sample_prob(self.v_out_prob) # dopuniti

        err = self.X2 - self.v_out_prob
        self.err_sum = tf.reduce_mean(err * err)

        self.initialize = tf.global_variables_initializer()

    def init_session(self):

        self.sess = tf.Session()
        self.sess.run(self.initialize)

    def train(self):

        for i in range(self.total_batch):
            batch, label = self.mnist.train.next_batch(self.batch_size)
            err, _ = self.sess.run([self.err_sum, self.out], feed_dict={self.X2: batch})

            if i % (int(self.total_batch / 10)) == 0:
                print(i, err)

        w1, w2, v_bias, h1_bias, h2_bias = self.sess.run([self.w1, self.w2, self.v_bias, self.h1_bias, self.h2_bias], feed_dict={self.X2: batch})
        v_out_prob, h_top = self.sess.run([self.v_out_prob, self.h2down], feed_dict={self.X2: self.test_x[0:self.Nu, :]})

        return w1, w2, v_bias, h1_bias, h2_bias, v_out_prob, h_top

    def visualize(self, w1, w2, v_bias, h1_bias, h2_bias, v_out_prob, h_top):

        # vizualizacija težina
        utils.draw_weights(w1, self.v_shape, self.Nh1, self.h1_shape, name="dbn_weights_w1.png")
        utils.draw_weights(w2, self.h1_shape, self.Nh2, self.h2_shape, name="dbn_weights_w2.png")

        # vizualizacija rekonstrukcije i stanja
        utils.draw_reconstructions(self.test_x, v_out_prob, h_top, self.v_shape, self.h2_shape, 20, name="dbn_reconstructions.png")

        self.reconstruct(0, h_top, self.test_x, w1, v_bias)  # prvi argument je indeks znamenke u matrici znamenki

        utils.draw_weights_freq(h_top, w1, self.v_shape, self.h2_shape, self.Nh2, name="dbn_weights_freq.png")

        r_input = self.generate_patterns()

        out = self.sess.run((self.v_out), feed_dict={self.h2up: r_input})

        # Emulacija dodatnih Gibbsovih uzorkovanja pomocu feed_dict
        for i in range(1000):
            out_prob, out, hout = self.sess.run((self.v_out_prob, self.v_out, self.h2down), feed_dict={self.X2: out})

        utils.draw_generated(r_input, hout, out_prob, self.v_shape, self.h2_shape, 50, name="dbn_generated.png")

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

    def reconstruct(self, ind, states, orig, weights, biases):

        # Slijedno iscrtavanje rekonstrukcije vidljivog sloja
        # ind - indeks znamenke u orig (matrici sa znamenkama kao recima)
        # states - vektori stanja ulaznih vektora
        # orig - originalnalni ulazni vektori
        # weights - matrica težina
        # biases - vektori pomaka vidljivog sloja

        j = 1
        in_a_row = 6
        Nimg = states.shape[1] + 3
        Nrows = int(np.ceil(float(Nimg + 2) / in_a_row))

        plt.figure(figsize=(12, 2 * Nrows))

        utils.draw_rec(states[ind], 'states', self.h2_shape, Nrows, in_a_row, j)
        j += 1
        utils.draw_rec(orig[ind], 'input', self.v_shape, Nrows, in_a_row, j)

        reconstr = biases.copy()
        j += 1
        utils.draw_rec(utils.sigmoid(reconstr), 'biases', self.v_shape, Nrows, in_a_row, j)

        for i in range(self.Nh2):
            if states[ind, i] > 0:
                j += 1
                reconstr = reconstr + weights[:, i]
                titl = '+= s' + str(i + 1)
                utils.draw_rec(utils.sigmoid(reconstr), titl, self.v_shape, Nrows, in_a_row, j)
        plt.tight_layout()


