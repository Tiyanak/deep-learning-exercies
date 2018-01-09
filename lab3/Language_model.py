import numpy as np
from lab3.RNN import RNN
from lab3.config import params

class LanguageModel():

    def run_language_model(self, dataset, max_epochs=5, hidden_size=100, sequence_length=30,
                           learning_rate=1e-1, sample_every=100):

        vocab_size = len(dataset.sorted_chars)
        rnn = RNN(hidden_size, sequence_length, vocab_size, learning_rate)  # initialize the recurrent network
        batch_size = params['batch_size']

        current_epoch = 0; batch = 0
        h0 = np.zeros((batch_size, hidden_size))

        while current_epoch < max_epochs:
            e, x, y = dataset.next_minibatch()

            if e:
                current_epoch += 1
                h0 = np.zeros((batch_size, hidden_size))

            x_oh, y_oh = self.oh(x, vocab_size), self.oh(y, vocab_size)
            loss, h0 = rnn.step(h0, x_oh, y_oh)

            if batch % sample_every == 0:
                conversation = "KRUGER:\nYou've made your point, Commissioner. There's only"
                smp = self.sample(dataset.encode(conversation), 100, rnn)
                print("\nepoch/batch/max batches={}/{}/{} -> loss={}".format(current_epoch + 1, (batch % dataset.num_batches) + 1, dataset.num_batches, loss))
                print("Predicted conversation: \n", conversation + dataset.decode(smp))

            batch += 1

    def sample(self, seed, n_sample, rnn):

        h0 = np.zeros((1, rnn.hidden_size))
        seed_onehot = self.oh(seed, rnn.vocab_size)
        sample = np.zeros((n_sample,))

        h, _ = rnn.rnn_forward(seed_onehot, h0, rnn.U, rnn.W, rnn.b)
        h = h.transpose(1, 0, 2)[-1]

        for i in range(0, n_sample):

            out = rnn.output(h[np.newaxis, :, :], rnn.V, rnn.c)
            yh = rnn.softmax(out)
            sample[i] = np.argmax(yh)

            x = np.zeros([1, rnn.vocab_size])
            x[0, (int)(sample[i])] = 1
            h, _ = rnn.rnn_step_forward(x, h, rnn.U, rnn.W, rnn.b)

        return sample

    def oh(self, x, vocab_size):

        if len(x.shape) == 1:
            x_oh = np.zeros([1, x.shape[0], vocab_size])
            for i in range(0, x.shape[0]):
                x_oh[0][i][x[i]] = 1
        else:
            x_oh = np.zeros([x.shape[0], x.shape[1], vocab_size])
            for i in range(0, x.shape[0]):
                for j in range(0, x.shape[1]):
                    x_oh[i][j][x[i][j]] = 1

        return x_oh