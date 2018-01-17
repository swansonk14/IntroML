from tqdm import trange

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

#-------------------------------------------------------------------------------
# Part 1 - Kernel Perceptron
#-------------------------------------------------------------------------------

class KernelPerceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, kernel=lambda x_j, x_i: np.dot(x_j, x_i), T=5):
        """Initializes the KernelPerceptron.

        Arguments:
            kernel(function): The kernel function to use.
            T(int): The number of iterations to perform in the Perceptron algorithm.
        """

        self.kernel = kernel
        self.T = T

    def fit(self, X, y):
        """Fits (trains) the KernelPerceptron on the provided data and labels.

        X(ndarray): An NxD numpy array where N is the number of data points
                    and D is the number of features (dimensions) of each data point.
        y(ndarray): A length N numpy array with the labels for each data point.

        Returns:
            The trained KernelPerceptron.
        """

        # Confirm valid input
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        # Save training data and labels for use in predict function
        self.X_ = X
        self.y_ = y

        # Initialize alpha
        n = X.shape[0]

        self.alpha = np.zeros(n)

        # Kernel perceptron algorithm
        for _ in range(self.T):
            for i in range(n):

                total = 0
                for j in range(n):
                    if self.alpha[j] != 0:
                        total += self.kernel(X[i], X[j]) * self.alpha[j] * y[j]

                if np.sign(total) != y[i]:
                    self.alpha[i] += 1

        return self

    def predict(self, X):
        """Predicts the labels for the provided data.

        Arguments:
            X(ndarray): An NxD numpy array where N is the number of data points
                        and D is the number of features (dimensions) of each data point.

        Returns:
            A length N numpy array containing the predicted labels for each data point.
        """

        # Confirm valid input
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)

        # Initialize predictions array
        n_ = self.X_.shape[0]
        n = X.shape[0]

        predictions = np.ones(n)

        # Generate predictions
        for i in trange(n):
            x = X[i]

            total = 0
            for j in range(n_):
                if self.alpha[j] != 0:
                    total += self.kernel(x, self.X_[j]) * self.alpha[j] * self.y_[j]

            predictions[i] = np.sign(total)

        return predictions
