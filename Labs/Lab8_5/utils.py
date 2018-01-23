import matplotlib.pyplot as plt
import numpy as np

def generate_3d_data(low, high, n, func):
    """Generates 3D data for a function.

    Arguments:
        low(int): The smallest value for x and y.
        high(int): The largest value for x and y.
        n(int): The number of data points to generate.
        func(function): The 3D function to apply.

    Returns:
        A tuple (X,y) where X is an n x 2 numpy array
        containing 2D data points and y is a length n
        numpy array containing the result of applying
        func to the data points in X.
    """

    a = np.random.uniform(low, high, n)
    b = np.random.uniform(low, high, n)
    c = func(a, b)

    X = np.c_[a, b]
    y = c

    return X, y

def plot_predictions_3d(ax_data, ax_pred, low, high, func, model):
    """Plots a 3D contour of the function c=sqrt(a^2 + b^2) and the model's predictions.

    Arguments:
        ax_data(Axes3D): A matplotlib Axes3D object.
        ax_pred(Axes3D): A matplotlib Axes3D object
        low(int): The smallest value for x and y.
        high(int): The largest value for x and y.
        func(function): The 3D function underlying the data.
        model(Sequential): A trained keras Sequential model.
    """

    # Generate mesh grid
    x = np.linspace(low, high, 100)
    y = np.linspace(low, high, 100)
    X, Y = np.meshgrid(x, y)
    
    # Plot the function
    Z = func(X, Y)
    ax_data.contour3D(X, Y, Z, 500, cmap=plt.cm.viridis)

    # Plot the model predictions
    Z_pred = model.predict(np.c_[X.ravel(), Y.ravel()])
    Z_pred = Z_pred.reshape(X.shape)
    ax_pred.contour3D(X, Y, Z_pred, 500, cmap=plt.cm.plasma)
