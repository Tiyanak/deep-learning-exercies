import tensorflow as tf
import matplotlib.pyplot as plt
from lab4 import utils
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class RBM():

    def __init__(self):

        self.init_params()
        self.init_data()

    def init_params(self):

        self.Nh = 100  # broj elemenata skrivenog sloja
        self.h_shape = (10, 10)  # 2D prikaz broja stanja
        self.Nv = 784  # broj elemenata vidljivog sloja
        self.v_shape = (28, 28)  # velicina slike
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

    def init_model(self):

        self.X = tf.placeholder("float", [None, 784])
        self.w = utils.weights([self.Nv, self.Nh])
        self.vb = utils.bias([self.Nv])
        self.hb = utils.bias([self.Nh])

        self.h0_prob = tf.nn.sigmoid(tf.matmul(self.X, self.w) + self.hb) # dopuniti
        self.h0 = utils.sample_prob(self.h0_prob)
        self.h = self.h0

        for step in range(self.gibbs_sampling_steps):
            self.v_prob = tf.nn.sigmoid(tf.matmul(self.h, tf.transpose(self.w)) + self.vb) # dopuniti
            self.v = utils.sample_prob(self.v_prob) # dopuniti
            self.h_prob = tf.nn.sigmoid(tf.matmul(self.v, self.w) + self.hb) # dopuniti
            self.h = utils.sample_prob(self.h_prob) # dopuniti

        self.w_positive_grad = tf.matmul(tf.transpose(self.X), self.h0_prob) # dopuniti
        self.w_negative_grad = tf.matmul(tf.transpose(self.v_prob), self.h_prob) # dopuniti

        self.dw = (self.w_positive_grad - self.w_negative_grad) / tf.to_float(tf.shape(self.X)[0])

        self.update_w = tf.assign_add(self.w, self.alpha * self.dw)
        self.update_vb = tf.assign_add(self.vb, self.alpha * tf.reduce_mean(self.X - self.v_prob, 0))
        self.update_hb = tf.assign_add(self.hb, self.alpha * tf.reduce_mean(self.h0 - self.h, 0))

        self.out = (self.update_w, self.update_vb, self.update_hb)

        self.v_prob = tf.nn.sigmoid(tf.matmul(self.h0, tf.transpose(self.w)) + self.vb) # dopuniti
        self.v = utils.sample_prob(self.v_prob) # dopuniti

        self.err = self.X - self.v_prob
        self.err_sum = tf.reduce_mean(self.err * self.err)

        self.initialize = tf.global_variables_initializer()

    def init_session(self):

        self.sess = tf.Session()
        self.sess.run(self.initialize)

    def set_init_weghts(self, w, bias, nh):

        self.w = w
        self.vb = bias
        self.Nh = nh
        self.init_model()

    def train(self):

        for i in range(self.total_batch):
            batch, label = self.mnist.train.next_batch(self.batch_size)
            err, _ = self.sess.run([self.err_sum, self.out], feed_dict={self.X: batch})

            if i % (int(self.total_batch / 10)) == 0:
                print(i, err)

        w = self.w.eval(session=self.sess)
        vb = self.vb.eval(session=self.sess)
        hb = self.hb.eval(session=self.sess)
        vr, hs = self.sess.run([self.v_prob, self.h], feed_dict={self.X: self.test_x[0: self.Nu, :]})

        return (w, vb, hb, vr, hs)

    def visualize(self, w, vb, vr, hs):

        # vizualizacija težina
        utils.draw_weights(w, self.v_shape, self.Nh, self.h_shape, name="weights.png")

        # vizualizacija rekonstrukcije i stanja
        utils.draw_reconstructions(self.test_x, vr, hs, self.v_shape, self.h_shape, 20, name="reconstructions.png")

        self.reconstruct(0, hs, self.test_x, w, vb)  # prvi argument je indeks znamenke u matrici znamenki

        self.draw_weights_freq(hs, w)

        r_input = self.generate_patterns()

        out = self.sess.run((self.v), feed_dict={self.h0: r_input})

        # Emulacija dodatnih Gibbsovih uzorkovanja pomocu feed_dict
        for i in range(1000):
            out_prob, out, hout = self.sess.run((self.v_prob, self.v, self.h), feed_dict={self.X: out})

        utils.draw_generated(r_input, hout, out_prob, self.v_shape, self.h_shape, 50, name="generated.png")

    def generate_patterns(self):

        # Generiranje uzoraka iz slucajnih vektora
        r_input = np.random.rand(100, self.Nh)
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

    def draw_weights_freq(self, h1s, w1s, name):

        # Vjerojatnost da je skriveno stanje ukljuceno kroz Nu ulaznih uzoraka
        plt.figure()
        tmp = (h1s.sum(0) / h1s.shape[0]).reshape(self.h_shape)
        plt.imshow(tmp, vmin=0, vmax=1, interpolation="nearest")
        plt.axis('off')
        plt.colorbar()
        plt.title('vjerojatnosti (ucestalosti) aktivacije pojedinih neurona skrivenog sloja')

        # Vizualizacija težina sortitranih prema ucestalosti
        tmp_ind = (-tmp).argsort(None)
        utils.draw_weights(w1s[:, tmp_ind], self.v_shape, self.Nh, self.h_shape, name="visualize_weights.png")
        plt.title('Sortirane matrice tezina - od najucestalijih do najmanje koristenih')
        plt.savefig(name)

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

        utils.draw_rec(states[ind], 'states', self.h_shape, Nrows, in_a_row, j)
        j += 1
        utils.draw_rec(orig[ind], 'input', self.v_shape, Nrows, in_a_row, j)

        reconstr = biases.copy()
        j += 1
        utils.draw_rec(utils.sigmoid(reconstr), 'biases', self.v_shape, Nrows, in_a_row, j)

        for i in range(self.Nh):
            if states[ind, i] > 0:
                j += 1
                reconstr = reconstr + weights[:, i]
                titl = '+= s' + str(i + 1)
                utils.draw_rec(utils.sigmoid(reconstr), titl, self.v_shape, Nrows, in_a_row, j)
        plt.tight_layout()


