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
    raise NotImplementedError

def load_toy_data(toy_data_path):
    """Loads the 2D toy dataset as numpy arrays.

    Arguments:
        toy_data_path(str): Path to the toy dataset .csv file.

    Returns:
        A tuple (features, labels) in which features is an Nx2 numpy
        matrix and labels is a length-N vector of +1/-1 labels.
    """
    raise NotImplementedError

def plot_toy_data(data, labels):
    """Plots the toy data in 2D.

    Arguments:
        data(ndarray): An Nx2 ndarray of points.
        labels(ndarray): A length-N vector of +1/-1 labels.
    """
    raise NotImplementedError
