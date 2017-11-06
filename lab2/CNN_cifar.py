import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import os
from lab2.cifar_readdata import unpickle, shuffle_data, plot_training_progress, draw_conv_filters, draw_image
import time
from data import Random2DGaussian

SAVE_DIR = "C:\\Users\\Igor Farszky\\PycharmProjects\\duboko\\duboko_ucenje\\lab2\\cifar\\results\\graphs"
SAVE_DIR_FILTERS = "C:\\Users\\Igor Farszky\\PycharmProjects\\duboko\\duboko_ucenje\\lab2\\cifar\\results\\filters"

class CNN_cifar:

    def __init__(self, num_input, num_classes, learning_rate=0.001, weight_decay=0.0001, pool_size=[3, 3], pool_strides=2):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_input = num_input
        self.num_classes = num_classes
        self.pool_size = pool_size
        self.pool_strides = pool_strides

        self.X = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.Y = tf.placeholder(tf.float32, [None, self.num_classes])

        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

        net = tf.layers.conv2d(self.X, filters=16, kernel_size=[5, 5], padding="same", name='conv1', activation=tf.nn.relu, kernel_regularizer=self.regularizer)
        net = tf.layers.max_pooling2d(net, pool_size=self.pool_size, strides=self.pool_strides, name='maxpool1')

        tf_conv1_graph = tf.get_default_graph()
        self.conv1_weights = tf_conv1_graph.get_tensor_by_name('conv1' + '/kernel:0')

        net = tf.layers.conv2d(net, filters=32, kernel_size=[5, 5], padding="same", name='conv2', activation=tf.nn.relu, kernel_regularizer=self.regularizer)
        net = tf.layers.max_pooling2d(net, pool_size=self.pool_size, strides=self.pool_strides, name='maxpool2')

        net = layers.flatten(net, scope='flatten')
        net = tf.layers.dense(net, units=256, name='fc1', activation=tf.nn.relu, kernel_regularizer=self.regularizer)
        net = tf.layers.dense(net, units=128, name='fc2', activation=tf.nn.relu, kernel_regularizer=self.regularizer)
        net = tf.layers.dense(net, units=10, name='fc3', activation=None, kernel_regularizer=self.regularizer)

        tf_graph = tf.get_default_graph()
        self.weights = tf_graph.get_tensor_by_name('fc3' + '/kernel:0')

        self.logits = net
        self.prediction = tf.nn.softmax(self.logits)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)) + weight_decay * tf.nn.l2_loss(self.weights)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.opt.minimize(self.loss)

        self.sess = tf.Session()

    def train(self, train_x, train_y, valid_x, valid_y, num_epochs=8, batch_size=50):

        self.sess.run(tf.global_variables_initializer())
        data = Random2DGaussian()

        plot_data = {}
        plot_data['train_loss'] = []
        plot_data['valid_loss'] = []
        plot_data['train_acc'] = []
        plot_data['valid_acc'] = []
        plot_data['lr'] = []

        num_examples = train_x.shape[0]
        num_batches = num_examples // batch_size

        for epoch_num in range(1, num_epochs + 1):
            train_x, train_y = shuffle_data(train_x, train_y)
            for step in range(num_batches):
                offset = step * batch_size
                # s ovim kodom pazite da je broj primjera djeljiv s batch_size
                batch_x = train_x[offset:(offset + batch_size), ...]
                batch_y = train_y[offset:(offset + batch_size)]
                feed_dict = {self.X: batch_x, self.Y: batch_y}
                start_time = time.time()
                run_ops = [self.train_op, self.loss, self.logits, self.conv1_weights]
                ret_val = self.sess.run(run_ops, feed_dict=feed_dict)
                _, loss_val, logits_val, conv1_weights = ret_val
                duration = time.time() - start_time
                if (step + 1) * batch_size % 2500 == 0:
                    sec_per_batch = float(duration)
                    format_str = 'epoch %d, step %d / %d, loss = %.2f (%.3f sec/batch)'
                    print(format_str % (epoch_num, (step + 1) * batch_size, num_batches * batch_size, loss_val, sec_per_batch))
                    draw_conv_filters(epoch=epoch_num, step=step, weights=conv1_weights, save_dir=SAVE_DIR_FILTERS)

            print("EPOCH STATISTICS : ")
            # train_loss, train_preds = self.sess.run([self.loss, self.prediction], feed_dict={self.X: train_x, self.Y: train_y})
            # train_acc, train_pr, train_m = data.eval_perf_multi(np.argmax(train_y, axis=1), np.argmax(train_preds, axis=1))
            # print("Train error: epoch {} loss={} accuracy={} precision={}".format(epoch_num, train_loss, train_acc, train_pr[0][0]))

            valid_loss, valid_preds = self.sess.run([self.loss, self.prediction], feed_dict={self.X: valid_x, self.Y: valid_y})
            valid_acc, valid_pr, valid_m = data.eval_perf_multi(np.argmax(valid_y, axis=1), np.argmax(valid_preds, axis=1))
            print("Validation error: epoch {} loss={} accuracy={} precision={}".format(epoch_num, valid_loss, valid_acc, valid_pr[0][0]))

            lr = self.learning_rate

            # plot_data['train_loss'] += [train_loss]
            plot_data['valid_loss'] += [valid_loss]
            # plot_data['train_acc'] += [train_acc]
            plot_data['valid_acc'] += [valid_acc]
            plot_data['lr'] += [lr]
            plot_training_progress(SAVE_DIR, plot_data)

    def predict(self, input):

        preds = self.sess.run(self.prediction, feed_dict={self.X: input})
        return preds
