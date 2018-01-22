from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.metrics import accuracy_score

#-------------------------------------------------------------------------------
# Part 6 - Neural Networks with Keras
#-------------------------------------------------------------------------------

def keras_nn(n_input, n_hidden, n_output):
    """Builds neural network with a single hidden layer for classification in keras.

    ReLU activation should be used for the hidden layer
    and softmax activation should be used for the output layer.

    Arguments:
        n_input(int): The number of units in the input layer.
        n_hidden(int): The number of units in the hidden layer.
        n_output(int): The number of units in the output layer.

    Returns:
        A keras model with a single hidden layer for classification.
    """

    raise NotImplementedError

class NN:
    """A class representing a neural network with a single hidden layer for classification."""

    def __init__(self,
                 n_input,
                 n_hidden,
                 n_output,
                 learning_rate=0.01):
        """Initializes the neural network.

        Arguments:
            n_input(int): The number of units in the input layer.
            n_hidden(int): The number of units in the hidden layer.
            n_output(int): The number of units in the output layer.
            learning_rate(float): The learning rate to use when performing
                                  gradient descent.
        """

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.learning_rate = learning_rate

        self.initialize_weights()

    def initialize_weights(self):
        """Initializes the weights of the network.

        The following variables should be initialized:
            self.W1 - weights connecting input to hidden layer
            self.b1 - bias for hidden layer
            self.W2 - weights connecting hidden to output layer
            self.b2 - bias for output layer
        """

        np.random.seed(0)
        self.W1 = np.random.randn(self.n_input, self.n_hidden)
        self.b1 = np.random.randn(self.n_hidden)
        self.W2 = np.random.randn(self.n_hidden, self.n_output)
        self.b2 = np.random.randn(self.n_output)

    def softmax(self, o):
        """Computes the softmax function for the matrix o.

        Arguments:
            o(ndarray): An n by n_output ndarray where n is the
                        number of data points and n_output is the
                        size of the output layer. Each row corresponds
                        to a data point and should be the input to
                        the output layer for that data point.

        Returns:
            An n by n_output ndarray where the softmax function
            has been applied to each row of o separately.
        """

        c = np.max(o)

        exp_scores = np.exp(o - c)
        P = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return P

    def feed_forward(self, X):
        """Runs the network on data points and outputs probabilities.

        Arguments:
            X(ndarray): An n by n_input ndarray where n is the number
                        of data points and n_input is the size of each
                        data point.

        Returns:
            An n by n_output ndarray where each row corresponds to a data
            point and contains a vector of length n_output with the
            probabilities of each class.
        """

        z = np.dot(X, self.W1) + self.b1
        a = np.tanh(z)
        o = np.dot(a, self.W2) + self.b2
        P = self.softmax(o)

        return P

    def predict(self, X):
        """Predicts the classes of data points.

        Arguments:
            X(ndarray): An n by n_input ndarray where n is the number
                        of data points and n_input is the size of each
                        data point.

        Returns:
            A length n numpy array containing the predicted class
            for each data point.
        """

        P = self.feed_forward(X)
        predictions = np.argmax(P, axis=1)

        return predictions

    def compute_accuracy(self, X, y):
        """Computes the accuracy of the network on data.

        Arguments:
            X(ndarray): An n by n_input ndarray where n is the number
                        of data points and n_input is the size of each
                        data point.
            y(ndarray): A length n numpy array containing the correct
                        classes for the data points.

        Returns:
            A float with the accuracy of the neural network.
        """

        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)

        return accuracy

#-------------------------------------------------------------------------------
# Part 1 - Computing Loss
#-------------------------------------------------------------------------------

    def compute_loss(self, X, y):
        """Computes the loss the network on the data.

        Arguments:
            X(ndarray): An n by n_input ndarray where n is the number
                        of data points and n_input is the size of each
                        data point.
            y(ndarray): A length n numpy array containing the correct
                        classes for the data points.

        Returns:
            A float with the loss of the neural network.
        """

        raise NotImplementedError

#-------------------------------------------------------------------------------
# Part 2 - Updating Weights
#-------------------------------------------------------------------------------

    def train_step(self, X, y):
        """Performs a batch gradient descent step to train the weights.

        Arguments:
            X(ndarray): An n by n_input ndarray where n is the number
                        of data points and n_input is the size of each
                        data point.
            y(ndarray): A length n numpy array containing the correct
                        classes for the data points.
        """

        # Reshape y to work with derivatives
        n = X.shape[0]
        Y = np.zeros((n, self.n_output))
        Y[range(n),y] = 1

        # YOUR CODE GOES HERE

        raise NotImplementedError

#-------------------------------------------------------------------------------
# Part 3 - Training
#-------------------------------------------------------------------------------

    def train(self, X, y, num_epochs, verbose=False):
        """Trains the neural network.

        Arguments:
            X(ndarray): An n by n_input ndarray where n is the number
                        of data points and n_input is the size of each
                        data point.
            y(ndarray): A length n numpy array containing the correct
                        classes for the data points.
            num_epochs(int): The number of epochs for which to train
                             the network (i.e. how many times to perform
                             an update step based on all the training
                             examples).
            verbose(bool): True to print out the loss and accuracy after
                           every epoch.
        """

        raise NotImplementedError
