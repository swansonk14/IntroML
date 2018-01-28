from keras.models import Sequential
from keras.layers import Dense

#-------------------------------------------------------------------------------
# Part 1.1 - Learning the Pythagorean Function
#-------------------------------------------------------------------------------

def build_single_layer_regression_model(n_hidden):
    """Builds a single layer regression neural network.

    Layers:
        - Dense, n_hidden neurons, tanh activation
        - Dense, 1 neuron, linear activation

    Arguments:
        n_hidden(int): The number of hidden units in
                       the hidden layer.

    Returns:
        A keras Sequential model with the single layer
        neural network.
    """

    model = Sequential()
    model.add(Dense(n_hidden, input_dim=2, activation='tanh'))
    model.add(Dense(1, activation='linear'))

    return model

#-------------------------------------------------------------------------------
# Part 2.2 - Power of Deep Networks
#-------------------------------------------------------------------------------

def build_deep_regression_model(n_hidden, n_layers):
    """Builds a deep regression neural network.

    Layers:
        - Dense, n_hidden neurons, tanh activation (x n_layers)
        - Dense, 1 neuron, linear activation

    Arguments:
        n_hidden(int): The number of hidden units in
                       each hidden layer.
        n_layers(int): The number of hidden layers (does
                       not include the output layer).

    Returns:
        A keras Sequential model with the deep neural network.
    """

    model = Sequential()
    model.add(Dense(n_hidden, input_dim=2, activation='tanh'))
    for _ in range(n_layers-1):
        model.add(Dense(n_hidden, activation='tanh'))
    model.add(Dense(1, activation='linear'))

    return model
