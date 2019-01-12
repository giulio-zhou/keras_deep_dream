from keras import optimizers
from keras.layers import Conv2D, Dense, Activation, Flatten
from keras.models import Model, Sequential
from keras.utils import to_categorical
from util import get_mnist_data

def simple_mlp(layer_n_units, activations, weights=[]):
   """
   layer_n_units: list containing number of units in each layer (length: N)
   activations: list of names of activations for each layer (length: N-1)
   weights: (optional) list of weights to initialize dense layers with.
   """
   assert len(layer_n_units) - 1 == len(activations)
   assert len(layer_n_units) >= 1
   assert len(weights) == 0 or len(weights) == len(layer_n_units) - 1
   weight_matrix = weights[0] if len(weights) > 0 else None
   model = Sequential() 
   model.add(Dense(units=layer_n_units[1], input_dim=layer_n_units[0],
                   weights=weight_matrix))
   model.add(Activation(activations[0]))
   for i in range(2, len(layer_n_units)):
       weight_matrix = weights[i-1] if len(weights) > 0 else None
       model.add(Dense(units=layer_n_units[i], weights=weight_matrix))
       model.add(Activation(activations[i-1]))
   return model

def train_mnist_mlp():
    layer_n_units = [784, 200, 10]
    activations = ['tanh', 'softmax']
    loss = 'categorical_crossentropy'
    optimizer = 'adam'
    metrics = ['accuracy']

    train_x, train_y, test_x, test_y = get_mnist_data()
    train_y, test_y = to_categorical(train_y), to_categorical(test_y)
    train_x, test_x = train_x.reshape(-1, 784), test_x.reshape(-1, 784)
    train_x, test_x = train_x / 255., test_x / 255.

    model = simple_mlp(layer_n_units, activations)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.fit(train_x, train_y, epochs=20, batch_size=128)

    print('=================')
    print(model.evaluate(test_x, test_y, batch_size=128))
    model.save('mnist_mlp.h5')

def simple_mnist_cnn():
    model = Sequential()
    model.add(Conv2D(8, 3, strides=(1, 1), padding="same",
                     input_shape=(28, 28, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(8, 3, strides=(2, 2), padding="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(16, 3, strides=(1, 1), padding="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(16, 3, strides=(2, 2), padding="same"))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(units=200))
    model.add(Activation('relu'))
    model.add(Dense(units=10))
    model.add(Activation('softmax'))
    return model

def train_mnist_simple_cnn():
    loss = 'categorical_crossentropy'
    optimizer = 'adam'
    metrics = ['accuracy'] 

    train_x, train_y, test_x, test_y = get_mnist_data()
    train_x = train_x.reshape(-1, 28, 28, 1)
    test_x = test_x.reshape(-1, 28, 28, 1)
    train_x, test_x = train_x / 255., test_x / 255.
    train_y, test_y = to_categorical(train_y), to_categorical(test_y)

    model = simple_mnist_cnn()
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.fit(train_x, train_y, epochs=20, batch_size=128)

    print('=================')
    print(model.evaluate(test_x, test_y, batch_size=128))
    model.save('mnist_simple_cnn.h5')

if __name__ == '__main__':
    train_mnist_mlp()
    train_mnist_simple_cnn()
