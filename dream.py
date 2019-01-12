from keras.models import load_model, Model
from keras.utils import to_categorical
from util import get_mnist_data
import keras.backend as K
import numpy as np
import skimage.io as skio
import sys

def apply_iterative_grads(model, data, grad_obj, lr=0.1, num_iters=20):
    grads = K.gradients(grad_obj, model.input)
    get_grad_fn = K.function([model.input], [grads[0]])
    curr_data = data
    for j in range(num_iters):
        grads = get_grad_fn([curr_data])
        dreamed_data = np.clip(curr_data + lr * grads[0], 0, 1)
        curr_data = dreamed_data
    return curr_data

def mnist_mlp(model_path):
    train_x, train_y, test_x, test_y = get_mnist_data()
    train_y, test_y = to_categorical(train_y), to_categorical(test_y)
    train_x, test_x = train_x.reshape(-1, 784), test_x.reshape(-1, 784)
    model = load_model(model_path)
    np.random.seed(1337)
    idx = np.random.choice(np.arange(len(test_x)), 10, replace=False)
    data, labels = test_x[idx] / 255., test_y[idx]
    outputs = model.layers[-2].output
    # Apply gradients for each class and concatenate.
    gradient_imgs = [np.hstack(data)]
    for i in range(10):
        curr_data = apply_iterative_grads(model, data, outputs[:, i],
                                          lr=0.1, num_iters=20)
        curr_data = map(lambda x: x.reshape(28, 28), curr_data)
        composite_img = np.hstack(curr_data)
        gradient_imgs.append(composite_img)
    composite_img = np.vstack(gradient_imgs)
    skio.imsave('mnist_mlp.png', composite_img)

def mnist_cnn(model_path):
    train_x, train_y, test_x, test_y = get_mnist_data()
    train_y, test_y = to_categorical(train_y), to_categorical(test_y)
    train_x = np.expand_dims(train_x, axis=-1)
    test_x = np.expand_dims(test_x, axis=-1)
    model = load_model(model_path)
    np.random.seed(1337)
    idx = np.random.choice(np.arange(len(test_x)), 9, replace=False)
    data, labels = test_x[idx] / 255., test_y[idx]
    outputs = model.layers[-2].output
    # Apply gradients for each class and concatenate.
    gradient_imgs = [np.hstack(data)]
    for i in range(10):
        curr_data = apply_iterative_grads(model, data, outputs[:, i],
                                          lr=0.1, num_iters=20)
        curr_data = curr_data.reshape(-1, 28, 28)
        composite_img = np.hstack(curr_data)
        gradient_imgs.append(composite_img)
    composite_img = np.vstack(gradient_imgs)
    skio.imsave('mnist_cnn.png', composite_img)

if __name__ == '__main__':
    mode = sys.argv[1]
    model_path = sys.argv[2]
    if mode == 'mnist_mlp':
        mnist_mlp(model_path)
    elif mode == 'mnist_cnn':
        mnist_cnn(model_path)
