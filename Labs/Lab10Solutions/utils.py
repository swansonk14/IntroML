from keras.datasets import mnist
import matplotlib
matplotlib.rc('font', size=20)
from matplotlib import offsetbox
import matplotlib.pyplot as plt
import numpy as np

def plot_digits(X, title=''):
    """Plots a selection of 64-dimensional (8x8) digits.

    Arguments:
        X(ndarray): An n by 64 numpy matrix where each row
                    contains the 64 pixels for an 8x8 grayscale
                    image of a digit.
        title(str): The title of the plot.
    """

    n_img_per_row = 20
    img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
    for i in range(n_img_per_row):
        ix = 10 * i + 1
        for j in range(n_img_per_row):
            iy = 10 * j + 1
            img[ix:ix + 8, iy:iy + 8] = X[i * n_img_per_row + j].reshape((8, 8))

    plt.imshow(img, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()

def plot_explained_variance_ratio(explained_variance_ratio):
    """Plots the percent of variance explained by the principle components of a PCA (non-cumulative and cumulative).

    Arguments:
        explained_variance_ratio(list): A list of the percent of variance that
                                        each principle component explains (sorted
                                        in descending order.)

    """

    ax = plt.subplot(211)
    ax.plot(range(len(explained_variance_ratio)), explained_variance_ratio, marker='o')
    ax.set_xlabel('Principle component number')
    ax.set_ylabel('Percent of variance explained')

    ax = plt.subplot(212)
    ax.plot(range(len(explained_variance_ratio)), np.cumsum(explained_variance_ratio), marker='o')
    ax.set_xlabel('Principle component number')
    ax.set_ylabel('Cumulative percent of variance explained')

    plt.show()

def plot_embedding(X, y, images, title=''):
    """Plots a 2-dimensional embedding of 64-dimensional (8x8) digits.

    Arguments:
        X(ndarray): An n by 2 numpy matrix where each row
                    contains the 2-dimensional embedding of a digit.
        y(ndarray): A length n numpy vector containing the labels (0-9)
                    for the digits.
        images(ndarray): An n by 64 numpy matrix where each row contains
                         the 64 pixels for an 8x8 grayscale image of a digit.
        title(str): The title of the plot.
    """

    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.gca()

    # Plot each 2-dimesional embedding of a digit with
    # text indicating the label (0-9) of that digit.
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 12})

    # Add selected images of the digits.
    if hasattr(offsetbox, 'AnnotationBbox'):
        shown_images = np.array([[1., 1.]])
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            
            # Don't show points that are too close
            if np.min(dist) < 4e-3:
                continue

            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)

    plt.xticks([]), plt.yticks([])
    plt.title(title)
    plt.show()

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

def plot_kmeans(X, y, kmeans, title=''):
    """Plot the decision boundaries for a K-Means classifier.

    Arguments:
        X(ndarray): An NxD numpy array where N is the number of data points
                    and D is the number of features (dimensions) of each data point.
        y(ndarray): A length N numpy array with the labels for each data point.
        kmeans(object): A trained sklearn kmeans classifier.
        title(string): The title of the plot.
    """

    # Plot digits
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 fontdict={'weight': 'bold', 'size': 12})

    # Create meshgrid
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    # Plot decision countours
    X = np.c_[xx.ravel(), yy.ravel()]
    Z = kmeans.predict(X)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, 12, cmap=plt.cm.Paired, alpha=0.6)

    # Plot centroids
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)

    plt.title(title)
    plt.show()

def load_mnist(noisy=False):
    """Loads MNIST data for an autoencoder.

    Arguments:
        noisy(bool): True to add random noise to the images.

    Returns:
        A tuple of train data, train labels, test data, test labels
        appropriately formatted for an autoencoder.
    """

    # Load digits data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Flatten digits into vector and divide by 255
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    # Add random noise
    if noisy:
        noise_factor = 0.5
        x_train = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
        x_test = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

        x_train = np.clip(x_train, 0., 1.)
        x_test = np.clip(x_test, 0., 1.)

    return x_train, y_train, x_test, y_test

def plot_reconstructions(original_imgs, reconstructed_imgs, n):
    """Plots a selection of MNIST digits and an autoencoder's reconstruction of those digits.

    Arguments:
        original_imgs(ndarray): An n by 784 numpy matrix containing the
                                original MNIST digit images.
        reconstructed_imgs(ndarray): An n by 784 numpy matrix containing
                                     the MNIST digit images reconstructed
                                     by an autoencoder.
        n(int): The number of images to display.
    """

    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i+1)
        ax.imshow(original_imgs[i].reshape(28, 28), cmap=plt.cm.gray)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, n+i+1)
        ax.imshow(reconstructed_imgs[i].reshape(28, 28), cmap=plt.cm.gray)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

def plot_generated_digits_vae(generator, epsilon_std):
    """Uses a trained variational autoencoder generator to generate digits.

    Arguments:
        generator(Model): A trained keras variational autoencoder generator
                          which can take a 1 by 2 numpy matrix and produce
                          a length 784 numpy vector representing a 28 by 28
                          image of a digit.
        epsilon_std(float): The standard deviation by which to multiple the
                            sample points before feeding them to the generator.
    """

    n = 15
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))

    # Sample n points within [-15, 15] standard deviations
    grid_x = np.linspace(-15, 15, n)
    grid_y = np.linspace(-15, 15, n)

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]]) * epsilon_std
            x_decoded = generator.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.imshow(figure, cmap=plt.cm.jet)
    plt.xticks([])
    plt.yticks([])
    plt.title('Digits generated by variational autoencoder')
    plt.show()
