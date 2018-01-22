# Partially based on http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/

import numpy as np
from sklearn.metrics import accuracy_score

class NN:
    """A class representing a single layer neural network for classification."""

    def __init__(self,
                 n_input,
                 n_hidden,
                 n_output,
                 learning_rate=0.01,
                 reg_lambda=0.01):
        """Initializes the neural network.

        Arguments:
            n_input(int): The number of units in the input layer.
            n_hidden(int): The number of units in the hidden layer.
            n_output(int): The number of units in the output layer.
            learning_rate(float): The learning rate to use when performing
                                  gradient descent.
            reg_lambda(float): The lambda regularization value.
        """

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda

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
        """Computes the softmax function for the array o.

        Arguments:
            o(ndarray): A length n_output numpy array representing
                        the input to the output layer.

        Returns:
            A length n_output numpy array with the result of applying
            the softmax function to o.
        """

        exp_scores = np.exp(o)
        probs = exp_scores / np.sum(exp_scores)

        return probs

    def feed_forward(self, x):
        """Runs the network on a data point and outputs probabilities.

        Arguments:
            x(ndarray): An length n_input numpy array where n_input is
                        the dimensionality of a data point.

        Returns:
            A length n_output numpy array containing the probabilities
            of each class according to the neural network.
        """

        z = np.dot(x, self.W1) + self.b1
        a = np.tanh(z)
        o = np.dot(a, self.W2) + self.b2
        probs = self.softmax(o)

        return probs

    def predict(self, x):
        """Predicts the class of a data point.

        Arguments:
            x(ndarray): An length n_input numpy array where n_input is
                        the dimensionality of a data point.

        Returns:
            A length n numpy array containing the predicted class
            for each data point.
        """

        probs = self.feed_forward(x)
        prediction = np.argmax(probs)

        return prediction

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

        predictions = np.array([self.predict(x) for x in X])
        accuracy = accuracy_score(y, predictions)

        return accuracy
