import time
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from lab2.tf_CNN import tf_CNN
from lab2.CNN_cifar import CNN_cifar
import os
from lab2.cifar_readdata import shuffle_data, unpickle, draw_image
from data import Random2DGaussian

DATA_DIR = 'C:\\Users\\Igor Farszky\\PycharmProjects\\duboko\\duboko_ucenje\\lab2\\train_dirs\\train_dir\\data_dir'
CIFAR_DATA_DIR = 'C:\\Users\\Igor Farszky\\PycharmProjects\\duboko\\duboko_ucenje\\lab2\\cifar'

img_width = 32
img_height = 32
num_channels = 3

config = {}
config['max_epochs'] = 8
config['batch_size'] = 50
config['lr_policy'] = {1:{'lr':1e-1}, 3:{'lr':1e-2}, 5:{'lr':1e-3}, 7:{'lr':1e-4}}

def zad3() :
    np.random.seed(int(time.time() * 1e6) % 2 ** 31)

    dataset = input_data.read_data_sets(DATA_DIR, one_hot=True)

    train_x = dataset.train.images
    train_x = train_x.reshape([-1, 28, 28, 1])
    train_y = dataset.train.labels

    valid_x = dataset.validation.images
    valid_x = valid_x.reshape([-1, 28, 28, 1])
    valid_y = dataset.validation.labels

    test_x = dataset.test.images
    test_x = test_x.reshape([-1, 28, 28, 1])
    test_y = dataset.test.labels

    train_mean = train_x.mean()
    train_x -= train_mean
    valid_x -= train_mean
    test_x -= train_mean

    CNN = tf_CNN(num_input=50, num_classes=10)
    CNN.train(train_x, train_y, num_epochs=8, batch_size=100)
    CNN.predict(test_x)

    slika = test_x[0, :, :, 0]
    slika2 = test_x[len(test_x) - 1, :, :, 0]
    plt.figure(1)
    plt.imshow(slika, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    plt.figure(2)
    plt.imshow(slika2, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    plt.show()

def zad4():

    data = Random2DGaussian()

    train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
    train_y = []
    for i in range(1, 6):
        subset = unpickle(os.path.join(CIFAR_DATA_DIR, 'data_batch_%d' % i))
        train_x = np.vstack((train_x, subset['data']))
        train_y += subset['labels']
    train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1)
    train_y = np.array(train_y, dtype=np.int32)

    subset = unpickle(os.path.join(CIFAR_DATA_DIR, 'test_batch'))
    test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1).astype(np.float32)
    test_y = np.array(subset['labels'], dtype=np.int32)
    test_yoh = data.class_to_onehot(test_y)

    valid_size = 5000
    train_x, train_y = shuffle_data(train_x, train_y)
    train_y = data.class_to_onehot(train_y)

    valid_x = train_x[:valid_size, ...]
    valid_y = train_y[:valid_size, ...]

    train_x = train_x[valid_size:, ...]
    train_y = train_y[valid_size:, ...]

    data_mean = train_x.mean((0, 1, 2))
    data_std = train_x.std((0, 1, 2))

    train_x = (train_x - data_mean) / data_std
    valid_x = (valid_x - data_mean) / data_std
    test_x = (test_x - data_mean) / data_std

    CNN = CNN_cifar(num_input=100, num_classes=10)
    CNN.train(train_x, train_y, valid_x, valid_y, num_epochs=8, batch_size=50)
    test_preds = CNN.predict(test_x)
    test_preds_classes = np.argmax(test_preds, axis=1)
    test_preds_maxes = [np.max(i) for i in test_preds]

    sorted_preds = list(reversed(sorted((e,i) for i,e in enumerate(test_preds_maxes))))
    netocni_indexi = []
    for i in sorted_preds:
        if test_preds_classes[i[1]] != test_y[i[1]]:
            netocni_indexi.append(i[1])

    iter = 0
    for key in netocni_indexi:
        if iter == 20:
            break

        net_index = key

        tocan_razred = test_y[net_index]
        pred_razred = test_preds_classes[net_index]
        print("predikcija={}, tocno={}".format(pred_razred, tocan_razred))

        class_preds = list(reversed(sorted((e,i) for i,e in enumerate(test_preds[net_index]))))

        iter_class_preds = 3
        print_classes = ''
        for key_inner in class_preds:
            if iter_class_preds == 0:
                break
            print_classes += 'razred={} vjerojatnost={} :: '.format(key_inner[1], key_inner[0])
            iter_class_preds -= 1
        print(print_classes)
        print("\n")
        iter += 1

        slika = test_x[net_index, :, :, :]
        plt.figure(iter)
        draw_image(slika, data_mean, data_std)

if __name__ == "__main__":

    # zad3()

    zad4()