from keras import backend as K
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib
matplotlib.rc('font', size=20)
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imsave
from tqdm import trange

def load_mnist_fc():
    """Loads MNIST data for a fully connected network.

    Returns:
        A tuple of training data, training labels,
        test data, test labels.
    """

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test

def load_mnist_conv():
    """Loads MNIST data for a convolutional network.

    Returns:
        A tuple of training data, training labels,
        test data, test labels.
    """

    img_rows, img_cols = 28, 28

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test

# https://github.com/philipperemy/keras-visualize-activations/blob/master/read_activations.py

def get_activation_maps(model, model_input):
    """Get the activations for a model given an input.

    Arguments:
        model(Sequential): A keras Sequential model.
        model_inputs(ndarray): A numpy array with an input to the model.

    Returns:
        A list of activation maps for each layer of the model.
    """

    # Get all layer outputs
    outputs = [layer.output for layer in model.layers]

    # Evaluation functions
    funcs = [K.function([model.input] + [K.learning_phase()], [out]) for out in outputs]

    # Get activations, setting learning phase to 0 for test mode
    activation_maps = [func([[model_input], 0.])[0] for func in funcs]

    return activation_maps

def display_activation_maps(model, model_input, input_image):
    """Displays model activations.

    Arguments:
        model(Sequential): A keras Sequential model.
        model_inputs(ndarray): A numpy array with an input to the model.
        input_image(ndarray): The model input in image format.
    """

    activation_maps = get_activation_maps(model, model_input)
    num_maps = len(activation_maps)
    layer_names = [layer.name for layer in model.layers]

    ax = plt.subplot(num_maps+1, 1, 1)
    ax.imshow(input_image, cmap=plt.cm.binary_r)
    ax.set_title('Input')

    for i, (activation_map, layer_name) in enumerate(zip(activation_maps, layer_names)):
        shape = activation_map.shape

        if len(shape) == 4:
            activations = np.hstack(np.transpose(activation_map[0], (2, 0, 1)))
        elif len(shape) == 2:
            # Try to make it square as much as possible
            activations = activation_map[0]
            num_activations = len(activations)

            # Too hard to display on screen
            if num_activations > 128:
                square_param = int(np.floor(np.sqrt(num_activations)))
                activations = activations[0: square_param * square_param]
                activations = np.reshape(activations, (square_param, square_param))
            else:
                activations = np.expand_dims(activations, axis=0)
        else:
            raise Exception('len(shape) = 3 has not been implemented.')
        
        ax = plt.subplot(num_maps+1, 1, i+2)
        ax.imshow(activations, cmap=plt.cm.jet)
        ax.set_title(layer_name)
    
    plt.show()
