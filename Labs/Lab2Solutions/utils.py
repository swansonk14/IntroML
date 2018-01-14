import csv

import numpy as np
import matplotlib.pyplot as plt

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

def load_toy_data(toy_data_path):
    """Loads the 2D toy dataset as numpy arrays.

    Arguments:
        toy_data_path(str): Path to the toy dataset .csv file.

    Returns:
        A tuple (features, labels) in which features is an Nx2 numpy
        matrix and labels is a length-N vector of +1/-1 labels.
    """
    
    labels, xs, ys = np.loadtxt(toy_data_path, delimiter='\t', unpack=True)
    data = np.vstack((xs, ys)).T

    return data, labels

def plot_toy_results(features, labels, theta, theta_0=0):
    """Plots the toy data in 2D along with the decision boundary.

    Arguments:
        features(ndarray): An Nx2 ndarray of features (points).
        labels(ndarray): A length-N vector of +1/-1 labels.
        theta(ndarray): A numpy array describing the linear classifier.
        theta_0(float): A real valued number representing the offset parameter.
    """

    # Plot the points with labels represented as colors
    plt.subplots()
    colors = ['b' if label == 1 else 'r' for label in labels]
    plt.scatter(features[:, 0], features[:, 1], s=40, c=colors)
    xmin, xmax = plt.axis()[:2]

    # Plot the decision boundary
    xs = np.linspace(xmin, xmax)
    ys = -(theta[0]*xs + theta_0) / (theta[1] + 1e-16)
    plt.plot(xs, ys, 'k-')

    # Show the plot
    plt.suptitle('Classified Toy Data')
    plt.show()

def plot_tune_results(Ts, train_accs, val_accs):
    """Plots classification accuracy on the training and validation data versus T value.

    Arguments:
        Ts(list): A list of the T values tried.
        train_accs(list): A list of the train accuracies for each parameter value.
        val_accs(list): A list of the validation accuracies for each parameter value.
    """

    # Put the data on the plot
    plt.subplots()
    plt.plot(Ts, train_accs, '-o')
    plt.plot(Ts, val_accs, '-o')

    # Make the plot presentable
    plt.suptitle('Classification Accuracy vs T')
    plt.legend(['train','val'], loc='upper right', title='Partition')
    plt.xlabel('T')
    plt.ylabel('Accuracy')
    plt.show()

def most_explanatory_words(theta, word_list):
    """Returns the word associated with the bag-of-words feature having largest weight.

    Arguments:
        theta(ndarray): A numpy array describing the linear classifier.
        word_list(list): A list of words in the order in which they appear as features.

    Returns:
        A list of words sorted by their weight in theta.
    """

    return [word for (theta_i, word) in sorted(zip(theta, word_list))[::-1]]
