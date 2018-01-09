import numpy as np
from lab3.config import params

class Dataset() :

    def __init__(self):

        self.batch_size = params['batch_size']
        self.sequence_length = params['sequence_length']

    def preprocess(self, input_file):

        with open(input_file, "r") as f:
            data = f.read()

        # count and sort most frequent characters
        data = list(data)
        datamap = {}
        for c in data:
            if c not in datamap:
                datamap[c] = 1
            else:
                datamap[c] += 1

        data_temp = sorted(datamap.items(), key=lambda item: (item[1], item[0]), reverse=True)
        self.sorted_chars = [x[0] for x in data_temp]

        # self.sorted chars contains just the characters ordered descending by frequency
        self.char2id = dict(zip(self.sorted_chars, range(len(self.sorted_chars))))
        # reverse the mapping
        self.id2char = {k:v for v,k in self.char2id.items()}

        # convert the data to ids
        data = np.array([self.char2id[x] for x in data])
        self.x = data[:-1]
        self.y = data[1:]

        self.num_batches = int(len(self.x) / (self.batch_size * self.sequence_length))
        self.current_batch = 0

    def encode(self, sequence):
        return np.array([self.char2id[x] for x in sequence])

    def decode(self, encoded_sequence):
        return "".join([self.id2char[x] for x in encoded_sequence])

    def next_minibatch(self):

        new_epoch = False

        start = self.current_batch * self.batch_size * self.sequence_length
        end = start + self.batch_size * self.sequence_length

        batch_x, batch_y = self.x[start:end], self.y[start:end]

        batch_x = batch_x.reshape(self.batch_size, self.sequence_length)
        batch_y = batch_y.reshape(self.batch_size, self.sequence_length)

        if self.current_batch >= self.num_batches - 1:
            new_epoch = True
            self.current_batch = 0
        else:
            self.current_batch += 1

        return new_epoch, batch_x, batch_y