# Partially based on http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html

import matplotlib
matplotlib.rc('font', size=20)
import matplotlib.pyplot as plt
import numpy as np

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in.

    Arguments:
        x(ndarray): A 1-dimensional numpy array representing
                    points on the x axis.
        y(ndarray): A 1-dimensional numpy array representing
                    points on the y axis.
        h(float): The step size of the meshgrid.

    Returns:
        A tuple of ndarrays (xx,yy) containing the mesh grid.
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_classifier(ax, X, y, clf, phi=lambda x: x, title=''):
    """Plot the decision boundaries for a classifier.

    Arguments:
        ax(object): A matplotlib axes object.
        X(ndarray): An NxD numpy array where N is the number of data points
                    and D is the number of features (dimensions) of each data point.
        y(ndarray): A length N numpy array with the labels for each data point.
        clf(object): A trained sklearn classifier.
        phi(function): A data transformation function.
        title(string): The title of the plot.
    """

    # Create meshgrid
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    # Plot decision countours
    X = np.c_[xx.ravel(), yy.ravel()]
    X_prime = np.array([phi(x) for x in X])
    Z = clf.predict(X_prime)
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.6)

    # Plot data
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=100, edgecolors='k')

    ax.set_title(title)

def plot_data(ax, X, y, title=''):
    """Plots data.

    Arguments:
        ax(object): A matplotlib axes object.
        X(ndarray): An NxD numpy array where N is the number of data points
                    and D is the number of features (dimensions) of each data point.
        y(ndarray): A length N numpy array with the labels for each data point.
        title(string): The title of the plot.
    """

    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=100, edgecolors='k')
    ax.set_title(title)
