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

def plot_decision_boundary(ax, X, y, nn, title='', multiclass=False):
    """Plot the decision boundaries for a classifier.

    Arguments:
        ax(object): A matplotlib axes object.
        X(ndarray): An NxD numpy array where N is the number of data points
                    and D is the number of features (dimensions) of each data point.
        y(ndarray): A length N numpy array with the labels for each data point.
        nn(NN): An NN object representing a trained neural network.
        title(string): The title of the plot.
        multiclass(bool): True if the predictions of the network are
                          multiclass, meaning each prediction will
                          be a numpy array of probabilities rather
                          than a single prediction value.
    """

    # Create meshgrid
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    # Plot decision countours
    X = np.c_[xx.ravel(), yy.ravel()]
    Z = nn.predict(X)
    if multiclass:
        Z = np.argmax(Z, axis=1)
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

def plot_loss_and_accuracy(history):
    """Plots the loss and accuracy of a keras model over time.

    Arguments:
        history(History): A keras history object.
    """

    ax1 = plt.subplot(211)
    ax1.plot(history.history['loss'])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    ax2 = plt.subplot(212)
    ax2.plot(history.history['acc'])
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
