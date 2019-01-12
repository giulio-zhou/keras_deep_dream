from mnist.loader import MNIST
import numpy as np

def get_mnist_data():
    mnist_data = MNIST('python-mnist/data')
    train_X, train_y = mnist_data.load_training()
    test_X, test_y = mnist_data.load_testing()
    train_X = np.array(train_X).reshape(-1, 28, 28)
    test_X = np.array(test_X).reshape(-1, 28, 28)
    train_y, test_y = np.array(train_y), np.array(test_y)
    return train_X, train_y, test_X, test_y
