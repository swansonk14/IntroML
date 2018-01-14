import csv

import matplotlib.pyplot as plt
import numpy as np

def load_reviews_data(reviews_data_path):
    """Loads the reviews dataset as a list of dictionaries.

    Arguments:
        reviews_data_path(str): Path to the reviews dataset .csv file.

    Returns:
        A list of dictionaries where each dictionary maps column name
        to value for a row in the reviews dataset. Numeric fields are
        converted to integers.
    """

    numeric_fields = {'sentiment', 'helpfulY', 'helpfulN'}

    data = []
    with open(reviews_data_path) as data_file:
        for datum in csv.DictReader(data_file, delimiter='\t'):
            data.append({field: int(value) if field in numeric_fields else value for field, value in datum.items()})

    return data

def plot_data(X, y):
    """Plots a set of 2D data points, colored according to their labels.

    Arguments:
        X(ndarray): An Nx2 numpy array containing N 2-dimensional data points.
        y(ndarray): A length-N numpy array containing the labels for the data
            points (+1 or -1).
    """

    plt.figure()
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired, edgecolors='k')
    plt.show()

def plot_linear_classifier(X, y, theta, theta_0, title='', subplot=111):
    """Plots a linear classifier with margins on top of the labeled 2D data.

    Also indicates support vectors.

    Arguments:
        X(ndarray): An Nx2 numpy array containing N 2-dimensional data points.
        y(ndarray): A length-N numpy array containing the labels for the data
            points (+1 or -1).
        theta(ndarray): A length-d numpy array describing a linear classifier.
        theta_0(float): A float describing the offset of the linear classifier.
        title(str): The title of the plot.
        subplot(int): A 3-digit integer indicating which subplot should be used.
    """

    a = -theta[0] / theta[1]
    b = -theta_0 / theta[1]
    margin = 1 / np.linalg.norm(theta)
    support_vectors = np.array([X[i] for i in range(len(X)) if y[i] * (np.dot(theta, X[i]) + theta_0) == 1])

    plot_classifier_with_margin(X, y, a, b, margin, support_vectors, title, subplot)

def plot_svm(X, y, clf, title='', subplot=111):
    """Plots an svm decision boundary with margins on top of the labeled 2D data.

    Also indicates support vectors.

    Arguments:
        X(ndarray): An Nx2 numpy array containing N 2-dimensional data points.
        y(ndarray): A length-N numpy array containing the labels for the data
            points (+1 or -1).
        clf(sklearn.svm.SVC): A trained svm classifier.
        title(str): The title of the plot.
        subplot(int): A 3-digit integer indicating which subplot should be used.
    """

    w = clf.coef_[0]
    a = -w[0] / w[1]
    b = -clf.intercept_[0] / w[1]
    margin = 1 / np.linalg.norm(clf.coef_)

    plot_classifier_with_margin(X, y, a, b, margin, clf.support_vectors_, title, subplot)

def plot_classifier_with_margin(X, y, a, b, margin, support_vectors, title='', subplot=111):
    """Plots a decision boundary with margins on top of the labeled 2D data.

    Also indicates support vectors.

    Arguments:
        X(ndarray): An Nx2 numpy array containing N 2-dimensional data points.
        y(ndarray): A length-N numpy array containing the labels for the data
            points (+1 or -1).
        a(float): The slope of the decision boundary.
        b(float): The y-intercept of the decision boundary.
        margin(float): The distance between the decision boundary and the margin.
        support_vectors(ndarray): A numpy array containing 2-dimensional vectors
            indicating the support vectors.
        title(str): The title of the plot.
        subplot(int): A 3-digit integer indicating which subplot should be used.
    """

    # Get the separating line
    xx = np.linspace(-4, 4)
    yy = a * xx + b

    # Setup plot
    plt.subplot(subplot)
    plt.title(title)

    # Plot the decision boundary, margins, and support vectors
    margin_down = yy - np.sqrt(1 + a ** 2) * margin
    margin_up = yy + np.sqrt(1 + a ** 2) * margin

    plt.plot(xx, yy, 'k-')
    plt.plot(xx, margin_down, 'k--')
    plt.plot(xx, margin_up, 'k--')

    if len(support_vectors) > 0:
        plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=80,
                    facecolors='none', zorder=10, edgecolors='k')

    # Plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired, edgecolors='k')
