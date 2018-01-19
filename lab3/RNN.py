import numpy as np

class RNN():


    def __init__(self, hidden_size=100, sequence_length=30, vocab_size=70, learning_rate=1e-1):
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate

        self.U = np.random.normal(size=[vocab_size, hidden_size], scale=1e-2)  # ... input projection
        self.W = np.random.normal(size=[hidden_size, hidden_size], scale=1e-2)  # ... hidden-to-hidden projection
        self.b = np.zeros([hidden_size, 1])  # ... input bias

        self.V = np.random.normal(size=[hidden_size, vocab_size], scale=1e-2)  # ... output projection
        self.c = np.zeros([vocab_size, 1])  # ... output bias

        self.memory_U, self.memory_W, self.memory_V = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V)
        self.memory_b, self.memory_c = np.zeros_like(self.b), np.zeros_like(self.c)

    def rnn_step_forward(self, x, h_prev, U, W, b):

        # x - input data (minibatch size x input dimension)
        # h_prev - previous hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)

        h_current = np.tanh(np.dot(W, h_prev.T) + np.dot(x, U).T + b).T

        # h_current - (minibatch_size x hidden_size)
        return h_current, (h_current, h_prev, x)

    def rnn_forward(self, x, h0, U, W, b):

        # x - input data for the whole time-series (minibatch size x sequence_length x input dimension)
        # h0 - initial hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)

        x = x.transpose(1, 0, 2); h = [h0]; cache = []

        for x_t in x:
            h_t, cache_t = self.rnn_step_forward(x_t, h[-1], U, W, b)
            cache.append(cache_t)
            h.append(h_t)

        # return - (minibatch size x sequence_length x hidden_size)
        return np.array(h[1:]).transpose(1, 0, 2), cache

    def rnn_step_backward(self, grad_next, cache):

        # grad_next - (minibatch size x hidden size)
        # h_current - (minibatch_size x hidden_size)
        # h_prev - (minibatch_size x hidden_size)
        # x - (minibatch size x input_dimension)

        h_current, h_prev, x = cache

        da = grad_next * (1 - h_current ** 2) # da - (minibatch_size x hidden_size)

        dh_t_minus_1 = np.dot(da, self.W) # dh_t_minus_1 - (minibatch_size x hidden_size)

        dW = np.dot(h_prev.T, da) # dW - (hidden_size x hidden_size)
        dU = np.dot(x.T, da) # dU - (input_dimension x hidden_size)
        db = np.sum(da, axis=0)[:, np.newaxis] # db - (hidden_size x 1)

        return dh_t_minus_1, dU, dW, db

    def rnn_backward(self, dh, cache):

        # dh - (batch_size x seq_len x hidden_size)

        dh = dh.transpose(1, 0, 2)
        dU = np.zeros_like(self.U)
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dh_t_plus_1 = np.zeros_like(dh[-1])

        for v_do, cache_i in reversed(list(zip(dh, cache))):
            dh_t = v_do + dh_t_plus_1
            dh_t_minus_1, dU_i, dW_i, db_i = self.rnn_step_backward(dh_t, cache_i)
            dh_t_plus_1 = dh_t_minus_1
            dU += dU_i; dW += dW_i; db += db_i

        return dU, dW, db

    def output(self, h, V, c):

        # h - (batch_size x sequence_length x hidden_size)
        # V - (hidden_size x vocab_size)
        # c - (vocab_size x 1)

        return np.dot(h, V) + c.T # output - (batch_size x sequence_length x vocab_size)

    def softmax(self, x):

        # x - (minibatch_size x sequence_length x vocab_size)

        e_x = np.exp(x)
        e_x_sum = np.sum(e_x, axis=2)[:,:,np.newaxis]
        return e_x / e_x_sum # softmax - (minibatch_size x sequence_length x vocab_size)

    def output_loss_and_grads(self, h, V, c, y):

        # h - (batch_size x sequence_length x hidden_size)
        # V - (hidden_size x vocab_size)
        # c - (vocab_size x 1)
        # y - (batch_size x sequence_length x hidden_size)

        o = self.output(h, V, c) # out - (batch_size x sequence_length x vocab_size)
        y_kappa = self.softmax(o) # yh - (minibatch_size x sequence_length x vocab_size)
        loss = -1 * np.mean(np.sum(y * np.log(y_kappa), axis=2)) # loss - scalar
        do = y_kappa - y # do - (minibatch_size x sequence_length x vocab_size)

        dV = np.zeros_like(V)
        dc = np.zeros_like(c)
        dh = []

        for do_t, h_t in zip(do.transpose(1, 0, 2), h.transpose(1, 0, 2)):
            dV += np.dot(h_t.T, do_t)
            dc = (dc.T + np.mean(do_t, axis=0)).T
            dh.append(np.dot(do_t, V.T))
        dh = np.array(dh).transpose(1, 0, 2)

        return loss, dh, dV, dc

    def update(self, dU, dW, db, dV, dc, delta=1e-6):

        dU = np.clip(dU, -5, 5)
        dW = np.clip(dW, -5, 5)
        db = np.clip(db, -5, 5)
        dV = np.clip(dV, -5, 5)
        dc = np.clip(dc, -5, 5)

        self.memory_U += (dU * dU)
        self.memory_W += (dW * dW)
        self.memory_b += (db * db)
        self.memory_V += (dV * dV)
        self.memory_c += (dc * dc)

        self.U -= (self.learning_rate / (delta + np.sqrt(self.memory_U))) * dU
        self.W -= (self.learning_rate / (delta + np.sqrt(self.memory_W))) * dW
        self.b -= (self.learning_rate / (delta + np.sqrt(self.memory_b))) * db
        self.V -= (self.learning_rate / (delta + np.sqrt(self.memory_V))) * dV
        self.c -= (self.learning_rate / (delta + np.sqrt(self.memory_c))) * dc

    def step(self, h, x, y):

        # h - (minibatch_size x hidden_size)
        # x - (minibatch_size x sequence_length x input_dimension)
        # y - (minibatch_size x sequence_length x input_dimension)

        h, cache = self.rnn_forward(x, h, self.U, self.W, self.b) # h (returned) - (batch_size x seq_len x hidden_size)
        loss, dh, dV, dc = self.output_loss_and_grads(h, self.V, self.c, y)
        dU, dW, db = self.rnn_backward(dh, cache)
        self.update(dU, dW, db, dV, dc, delta=1e-7)

        return loss, h[:, -1, :]
