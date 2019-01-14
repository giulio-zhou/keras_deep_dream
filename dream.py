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

def get_dataset(name, is_mlp):
    if name == 'mnist':
        return MNIST(is_mlp)

class MNIST:
    def __init__(self, is_mlp=False):
        self.train_x, self.train_y, self.test_x, self.test_y = get_mnist_data()
        self.train_y = to_categorical(self.train_y)
        self.test_y = to_categorical(self.test_y)
        self.train_x, self.test_x = self.train_x / 255., self.test_x / 255.
        if is_mlp:
            self.train_x = self.train_x.reshape(-1, 784)
            self.test_x = self.test_x.reshape(-1, 784)
        else:
            self.train_x = np.expand_dims(self.train_x, axis=-1)
            self.test_x = np.expand_dims(self.test_x, axis=-1)
        self.is_mlp = is_mlp
        self.dims = (28, 28)

class SingleFeature:
    def __init__(self, model, feature, config, height=None, width=None):
        self.model = model
        self.feature = feature
        if hasattr(feature, 'inherent_dimensions'):
            self.height, self.width = feature.inherent_dimensions
            results = []
            for i in range(self.height):
                row_results = []
                for j in range(self.width):
                    set_config = self.feature.make_config(i, j, config)
                    result = run(model, set_config)[0]
                    result = result.reshape(*set_config[DIMS])
                    row_results.append(result)
                results.append(row_results)
            output = np.vstack(map(np.hstack, results))
        else:
            self.height, self.width = height, width
            results = []
            for i in range(len(self.feature)):
                set_config = self.feature.make_config(i, config)
                result = run(model, set_config)[0]
                result = result.reshape(*set_config[DIMS])
                results.append(result)
            rows = np.split(np.hstack(results), height)
            output = np.vstack(rows)
        self.output = output

class DoubleFeature:
    def __init__(self, model, features, config):
        self.model = model
        self.feature1, self.feature2 = features
        self.height, self.width = len(self.feature1), len(self.feature2)
        results = []
        for i in range(self.height):
            row_results = []
            for j in range(self.width):
                set_config = self.feature1.make_config(i, config)
                set_config = self.feature2.make_config(j, set_config)
                result = run(model, set_config)
                result = result.reshape(*set_config[DIMS])
                row_results.append(result)
            results.append(row_results)
        self.output = np.vstack(map(np.hstack, results))

def run(model, config):
    data = config[DATA]
    grad_obj = config[LOSS_TENSOR]
    lr, num_iters = config[LR], config[NUM_ITERS]
    dreamed_data = apply_iterative_grads(model, data, grad_obj, lr, num_iters)
    return dreamed_data

LR = 'learning_rate'
NUM_ITERS = 'num_iters'
DATA = 'data'
DIMS = 'dims'
LOSS_TENSOR = 'loss_tensor'

default_config = {
    LR: 0.01,
    NUM_ITERS: 20,
}

class LearningRates:
    def __init__(self, learning_rates):
        self.learning_rates = learning_rates
    def __len__(self):
        return len(self.learning_rates)
    def make_config(self, idx, config):
        config[LR] = self.learning_rates[idx]
        return config

class NumberOfIterations:
    def __init__(self, num_iters):
        self.num_iters = num_iters 
    def __len__(self):
        return len(self.num_iters)
    def make_config(self, idx, config):
        config[NUM_ITERS] = self.num_iters[idx]
        return config

class MultipleExamples:
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def make_config(self, idx, config):
        config[DATA] = [self.data[idx]]
        return config

# These contain gradient sources.
class AllIndividualNeurons:
    def __init__(self, model, layer_no):
        self.model = model
        self.layer_no = layer_no
        self.layer_obj = self.model.layers[self.layer_no].output
    def __len__(self):
        return self.layer_obj.get_shape()[1]
    def make_config(self, idx, config):
        config[LOSS_TENSOR] = self.layer_obj[:, idx]
        return config

class AllChannels:
    def __init__(self, model, layer_no):
        self.model = model
        self.layer_no = layer_no
        self.layer_obj = self.model.layers[self.layer_no].output
    def __len__(self):
        print(self.layer_obj, self.layer_obj.get_shape())
        return self.layer_obj.get_shape()[3]
    def make_config(self, idx, config):
        features = self.layer_obj[:, :, :, idx]
        config[LOSS_TENSOR] = K.sum(K.square(features), axis=[1, 2])
        return config

class AllSpatialColumns:
    def __init__(self, model, layer_no):
        self.model = model
        self.layer_no = layer_no
        self.layer_obj = self.model.layers[self.layer_no].output
        self.inherent_dimensions = self.layer_obj.get_shape()[1:3]
    def make_config(self, y_idx, x_idx, config):
        features = self.layer_obj[:, y_idx, x_idx]
        config[LOSS_TENSOR] = K.sum(K.square(features), axis=1)
        return config

def viz_mlp(model_path, dataset_name):
    dataset = get_dataset(dataset_name, is_mlp=True)
    model = load_model(model_path)
    np.random.seed(1337)
    idx = np.random.choice(np.arange(len(dataset.test_x)), 10, replace=False)
    data, labels = dataset.test_x[idx], dataset.test_y[idx]
    # Set configuration and create features.
    config = default_config
    config[DATA], config[DIMS] = [data[0]], dataset.dims
    config[NUM_ITERS] = 50
    feature = AllIndividualNeurons(model, -2)
    feature2 = LearningRates([1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1])
    # runner = SingleFeature(model, feature, config, height=2, width=5)
    runner = DoubleFeature(model, [feature, feature2], config)
    skio.imsave('%s_mlp.png' % dataset_name, runner.output)

def viz_cnn(model_path, dataset_name):
    dataset = get_dataset(dataset_name, is_mlp=False)
    model = load_model(model_path)
    np.random.seed(1337)
    idx = np.random.choice(np.arange(len(dataset.test_x)), 10, replace=False)
    data, labels = dataset.test_x[idx], dataset.test_y[idx]
    # Set configuration and create features.
    config = default_config
    config[DATA], config[DIMS] = [data[0]], dataset.dims
    config[NUM_ITERS] = 20
    # feature = AllSpatialColumns(model, 7)
    # runner = SingleFeature(model, feature, config)
    config[LR] = 2e-2
    feature = AllChannels(model, 7)
    # feature2 = LearningRates([1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1])
    feature2 = MultipleExamples(data)
    runner = DoubleFeature(model, [feature, feature2], config)
    skio.imsave('%s_cnn.png' % dataset_name, runner.output)

if __name__ == '__main__':
    mode = sys.argv[1]
    model_path = sys.argv[2]
    dataset_name = sys.argv[3]
    if mode == 'viz_mlp':
        viz_mlp(model_path, dataset_name)
    elif mode == 'viz_cnn':
        viz_cnn(model_path, dataset_name)
