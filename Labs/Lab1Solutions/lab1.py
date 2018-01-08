"""Utility functions for loading and plotting data."""

import csv
import numpy as np
import matplotlib.pyplot as plt

def load_reviews_data(reviews_data_path):
    """Loads the reviews dataset as a list of dictionaries.

    Arguments:
        reviews_data_path(str): Path to the reviews dataset .csv file.

    Returns:
        A list of dictionaries where each dictionary maps column name
        to value for a row in the reviews dataset.
    """

    with open(reviews_data_path) as data_file:
        data = list(csv.DictReader(data_file, delimiter='\t'))

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

def plot_toy_data(data, labels):
    """Plots the toy data as a 2D scatterplot.

    Data points with a +1 label should be blue.
    Data points with a -1 label should be red.

    Arguments:
        data(ndarray): An Nx2 ndarray of points.
        labels(ndarray): A length-N vector of +1/-1 labels.
    """

    # Plot points
    xs = data[:,0]
    ys = data[:,1]
    colors = ['b' if label == 1 else 'r' for label in labels]
    plt.scatter(xs, ys, s=40, c=colors)

    # Plot linear decision boundary
    xx = np.linspace(-4, 4)
    yy = -xx + 2.25
    plt.plot(xx, yy, 'k-')

    plt.show()
