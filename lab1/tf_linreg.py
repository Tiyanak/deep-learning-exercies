import tensorflow as tf

class Tf_linreg:

    def __init__(self, param_niter=100, learning_rate=0.1, param_lambda=1):
        self.param_niter = param_niter
        self.learning_rate = learning_rate
        self.param_lambda = param_lambda

    def train(self, Xin, Y_in):
        X = tf.placeholder(tf.float32, [None])
        Y_ = tf.placeholder(tf.float32, [None])
        a = tf.Variable(0.0)
        b = tf.Variable(0.0)

        Y = a * X + b

        loss = (Y - Y_) ** 2

        trainer = tf.train.GradientDescentOptimizer(self.learning_rate)
        train_op = trainer.minimize(loss)
        grad = trainer.compute_gradients(loss)

        gradijenti = []
        for sgrad, var in grad:
            gradijenti.append(tf.Print(sgrad, [sgrad], var.name))
            
        train_op = trainer.apply_gradients(grad)

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        for i in range(self.param_niter):
            val_loss, _, val_a, val_b = sess.run([loss, train_op, a, b], feed_dict={X: Xin, Y_: Y_in})
            print(i, val_loss, val_a, val_b)

        return (val_a, val_b)