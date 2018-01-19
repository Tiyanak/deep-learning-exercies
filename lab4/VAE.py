from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from lab4 import utils
import os
import numpy as np
import matplotlib.pyplot as plt

class VAE():

    def __init__(self):

        self.init_params()
        self.init_data()

    def init_data(self):

        self.mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True)
        self.n_samples = self.mnist.train.num_examples
        self.total_batch = int(self.n_samples / self.batch_size)

    def init_params(self):

        self.learning_rate = 0.001
        self.batch_size = 100
        self.n_epochs = 10

        self.n_hidden_recog_1 = 200  # 1 sloj enkodera
        self.n_hidden_recog_2 = 200  # 2 sloj enkodera
        self.n_hidden_gener_1 = 200  # 1 sloj dekodera
        self.n_hidden_gener_2 = 200  # 2 sloj dekodera

        self.n_z = 20  # broj skrivenih varijabli
        self.n_input = 784  # MNIST data input (img shape: 28*28)
        self.in_shape = (28, 28)

    def vae_layer(self, input_tensor, input_dim, output_dim, layer_name, act=tf.nn.softplus):

        """Kreiranje jednog skrivenog sloja"""
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            weights = utils.weight_variable([input_dim, output_dim], layer_name + '/weights')
            utils.variable_summaries(weights, 'weights')
            tf.summary.tensor_summary('weightsT', weights)
            biases = utils.bias_variable([output_dim])
            utils.variable_summaries(biases, 'biases')
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)

        return activations

    def create_session(self):

        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()

    def init_model(self):

        # definicije ulaznog tenzora
        self.x = tf.placeholder("float", [None, self.n_input]) # dopuniti

        # definirajte enkoiderski dio
        self.layer_e1 = self.vae_layer(self.x, self.n_input, self.n_hidden_recog_1, 'layer_e1')
        self.layer_e2 = self.vae_layer(self.layer_e1, self.n_hidden_recog_1, self.n_hidden_recog_2, 'layer_e2') # dopuniti

        with tf.name_scope('z'):
            # definirajte skrivene varijable i pripadajuci generator šuma
            self.z_mean = self.vae_layer(self.layer_e2, self.n_hidden_recog_2, self.n_z, 'z_mean', act=tf.identity)
            self.z_sigma = self.vae_layer(self.layer_e2, self.n_hidden_recog_2, self.n_z, 'z_sigma', act=tf.identity)
            self.z_sigma_sq = tf.square(self.z_sigma)
            self.z_log_sigma_sq = tf.log(self.z_sigma_sq)
            self.eps = tf.random_normal((self.batch_size, self.n_z), 0, 1, dtype=tf.float32)

            self.z = tf.add(self.z_mean, tf.multiply(self.z_sigma, self.eps))
            tf.summary.histogram('activations', self.z)

        # definirajte dekoderski dio
        self.layer_d1 = self.vae_layer(self.z, self.n_z, self.n_hidden_gener_1, 'layer_d1')
        self.layer_d2 = self.vae_layer(self.layer_d1, self.n_hidden_gener_1, self.n_hidden_gener_2, 'layer_d2') # dopuniti

        # definirajte srednju vrijednost rekonstrukcije
        self.x_reconstr_mean = self.vae_layer(self.layer_d2, self.n_hidden_gener_2, self.n_input, 'vae_out', act=tf.identity) # dopuniti

        self.x_reconstr_mean_out = tf.nn.sigmoid(self.x_reconstr_mean)

        # definirajte dvije komponente funkcije cijene
        with tf.name_scope('cost'):
            self.cost1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x, logits=self.x_reconstr_mean)  # dopuniti - E[log(p(x|z))]
            tf.summary.histogram('cross_entropy', self.cost1)
            self.cost2 = -0.5 * (1 + self.z_log_sigma_sq - tf.square(self.z_mean) - self.z_sigma_sq) # dopuniti - -Dkl(q(z|x)||p(z))
            tf.summary.histogram('D_KL', self.cost2)
            self.cost = tf.reduce_mean(tf.reduce_sum(self.cost1, 1) + tf.reduce_sum(self.cost2, 1))  # average over batch
            tf.summary.histogram('cost', self.cost)

        # ADAM optimizer
        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # Prikupljanje podataka za Tensorboard
        self.merged = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self.train_writer = tf.summary.FileWriter('train', self.sess.graph)

    def init_session(self):

        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    def train(self):

        step = 0
        for epoch in range(self.n_epochs):
            avg_cost = 0.

            for i in range(self.total_batch):
                batch_xs, _ = self.mnist.train.next_batch(self.batch_size)

                opt, z_mean, z_sigma, z, x_reconstr_mean_out, cost1, cost2, cost = \
                    self.sess.run((self.optimizer, self.z_mean, self.z_sigma, self.z,
                                   self.x_reconstr_mean_out, self.cost1, self.cost2, self.cost), feed_dict={self.x: batch_xs})

                avg_cost += cost / self.n_samples * self.batch_size

            # Display logs per epoch step
            if epoch % (int(self.n_epochs / 10)) == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = self.sess.run([self.merged, self.optimizer], feed_dict={self.x: batch_xs},
                                           options=run_options, run_metadata=run_metadata)
                self.train_writer.add_run_metadata(run_metadata, 'epoch%03d' % epoch)
                self.train_writer.add_summary(summary, i)
                self.saver.save(self.sess, os.path.join('train', "model.ckpt"), epoch)

        self.train_writer.close()

    def visualize(self):

        self.visualize_reconstruction_and_state()
        self.visualize_test_sample_hidden()
        # self.fill_results_for_grid()

    # vizualizacija rekonstrukcije i stanja
    def visualize_reconstruction_and_state(self):

        x_sample = self.mnist.test.next_batch(100)[0]
        x_reconstruct, z_out = self.sess.run([self.x_reconstr_mean_out, self.z], feed_dict={self.x: x_sample})
        utils.draw_reconstructions_vae(x_sample, x_reconstruct, z_out, (28, 28), (4, 5), name="vae_reconstructions.png")

    # Vizualizacija raspored testnih uzoraka u 2D prostoru skrivenih varijabli - 1. nacin
    def visualize_test_sample_hidden(self):

        x_sample, y_sample = self.mnist.test.next_batch(5000)
        z_mu, z_sigma = self.sess.run((self.z_mean, self.z_log_sigma_sq), feed_dict={self.x: x_sample})

        utils.plot_latent(z_mu, y_sample)
        utils.save_latent_plot('vae_trt.png')

        fig = plt.figure(figsize=(15, 4)) # Vizualizacija ugašenih elemenata skrivenog sloja - 1. nacin
        utils.boxplot_vis(fig, 1, z_mu, 'Z mean values', 'Z elemets') # Vizualizacija statistike za z_mean
        self.visualize_stat_z_sigma(fig, z_sigma) # Vizualizacija statistike za z_sigma
        self.visualize_weight_stat_decoder_input(fig, z_mu) # Vizualizacija statistike za težine ulaza u dekoder
        plt.savefig("vae_visualize_stats.png")

    # Vizualizacija statistike za z_sigma
    def visualize_stat_z_sigma(self, fig, z_sigma):

        ax = utils.boxplot_vis(fig, 2, np.square(np.exp(z_sigma)), 'Z sigma values', 'Z elemets')
        ax.set_xlim([-0.05, 1.1])

    # Vizualizacija statistike za težine ulaza u dekoder
    def visualize_weight_stat_decoder_input(self, fig, z_mu):

        test = tf.get_default_graph().get_tensor_by_name("layer_d1/weights:0")
        weights_d1 = test.eval(session=self.sess)
        utils.boxplot_vis(fig, 3, weights_d1.T, 'Weights to decoder', 'Z elemets')

        # Vizualizacija ugašenih elemenata skrivenog sloja - 2. nacin
        # self.visualize_off_hidden(z_mu, weights_d1)

    # Vizualizacija ugašenih elemenata skrivenog sloja - 2. nacin
    def visualize_off_hidden(self, z_mu, weights_d1):

        fig = plt.figure(figsize=(15, 7))

        # 3D bar graf za z_mean
        labels = ('Samples', 'Hidden elements', 'Z mean')
        utils.bargraph_vis(fig, 1, z_mu, [200, z_mu.shape[1]], 'g', labels)

        # 3D bar graf za težine iz z_mena u dekoder
        labels = ('Decoder elements', 'Hidden elements Z', 'Weights')
        utils.bargraph_vis(fig, 2, weights_d1.T, weights_d1.T.shape, 'y', labels)

    def fill_results_for_grid(self):

        # Vizualizacija raspored testnih uzoraka u 2D prostoru skrivenih varijabli - 2. nacin
        nx = ny = 21
        x_values = np.linspace(-3, 3, nx)
        y_values = np.linspace(-3, 3, ny)

        canvas = np.empty((28 * ny, 28 * nx))

        Xi, Yi = np.meshgrid(x_values, y_values)
        Z = np.column_stack((Xi.flatten(), Yi.flatten()))
        X = np.empty((0, 28 * 28))
        ind = list(range(self.batch_size, nx * ny, self.batch_size))
        for i in np.array_split(Z, ind):
            if i.shape[0] < self.batch_size:
                i = np.concatenate((i, np.zeros((self.batch_size - i.shape[0], i.shape[1]))), 0)
            X = np.vstack((X, self.sess.run(self.x_reconstr_mean_out, feed_dict={self.z: i})))

        for i, yi in enumerate(y_values):
            for j, xi in enumerate(x_values):
                canvas[(nx - i - 1) * 28:(nx - i) * 28, j * 28:(j + 1) * 28] = X[i * nx + j].reshape(28, 28)

        plt.figure(figsize=(8, 10))
        plt.imshow(canvas, origin="upper")
        plt.xticks(np.linspace(14, 588 - 14, 11), np.round(np.linspace(-3, 3, 11), 2))
        plt.yticks(np.linspace(14, 588 - 14, 11), np.round(np.linspace(3, -3, 11), 2))
        plt.xlabel('z0')
        plt.ylabel('z1')
        plt.tight_layout()

