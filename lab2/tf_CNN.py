import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import os

class tf_CNN:

    def __init__(self, num_input, num_classes, learning_rate=0.001, weight_decay=1, pool_size=[2, 2], pool_strides=2):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_input = num_input
        self.num_classes = num_classes
        self.pool_size = pool_size
        self.pool_strides = pool_strides

        self.X = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.Y = tf.placeholder(tf.float32, [None, self.num_classes])

        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

        net = tf.layers.conv2d(self.X, filters=16, kernel_size=[5, 5], padding="same", name='conv1', activation=tf.nn.relu, kernel_regularizer=self.regularizer)
        net = tf.layers.max_pooling2d(net, pool_size=self.pool_size, strides=self.pool_strides, name='maxpool1')

        net = tf.layers.conv2d(net, filters=32, kernel_size=[5, 5], padding="same", name='conv2', activation=tf.nn.relu, kernel_regularizer=self.regularizer)
        net = tf.layers.max_pooling2d(net, pool_size=self.pool_size, strides=self.pool_strides, name='maxpool2')

        net = layers.flatten(net, scope='flatten')
        net = layers.fully_connected(net, num_outputs=512, scope='fc1', activation_fn=tf.nn.relu, weights_regularizer=self.regularizer)
        net = layers.fully_connected(net, num_outputs=10, scope='fc2', activation_fn=None, weights_regularizer=self.regularizer)

        self.weights = tf.get_default_graph()
        self.weights = self.weights.get_tensor_by_name(os.path.split(net.name)[0] + '/weights:0')

        self.logits = net
        self.prediction = tf.nn.softmax(self.logits)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)) + weight_decay * tf.nn.l2_loss(self.weights)
        self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = self.opt.minimize(self.loss)

        self.correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.sess = tf.Session()

    def train(self, train_x, train_y, num_epochs=8, batch_size=50):

        self.sess.run(tf.global_variables_initializer())

        num_examples = train_x.shape[0]
        num_batches = num_examples // batch_size

        for epoch in range(1, num_epochs + 1):

            permutation_idx = np.random.permutation(num_examples)
            train_x = train_x[permutation_idx]
            train_y = train_y[permutation_idx]

            for i in range(num_batches):

                batch_x = train_x[i * batch_size:(i + 1) * batch_size, :]
                batch_y = train_y[i * batch_size:(i + 1) * batch_size, :]

                _, loss, acc, weights = self.sess.run([self.train_op, self.loss, self.accuracy, self.weights], feed_dict={
                    self.X: batch_x, self.Y: batch_y
                })

                if i % 5 == 0:
                    print("epoch %d, step %d/%d, batch loss = %.2f" % (epoch, i * batch_size, num_examples, loss))
                if i > 0 and i % 50 == 0:
                    print("Train accuracy = %.2f" % acc)
            print("Train accuracy = %.2f" % acc)
            print("weghts: ", weights)
            # evaluate("Validation", valid_x, valid_y, net, loss, config)

    def predict(self, input):

        preds = self.sess.run(self.prediction, feed_dict={self.X: input})
        print("predictions={}".format(np.argmax(preds, axis=1)))
        return preds
