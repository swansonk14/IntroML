from keras import backend as K
from keras.metrics import binary_crossentropy
from keras.layers import Dense, Input, Lambda
from keras.models import Model
import numpy as np
from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics.pairwise import euclidean_distances

#-------------------------------------------------------------------------------
# Part 1.2.1 - PCA from scratch
#-------------------------------------------------------------------------------

class PCA:
    def __init__(self, n_components):
        """Initializes the PCA model.

        Arguments:
            n_components(int): The number of principal components
                               to use when transforming data.
        """

        self.n_components = n_components

    def fit(self, X):
        """Learns the transformation matrix self.Q for PCA.

        Arguments:
            X(ndarray): An n by d numpy array representing n
                        data points, each with d dimensions.

        Returns:
            The fitted PCA model.
        """

        raise NotImplementedError

    def transform(self, X):
        """Transforms the data into the top n_components principal components.

        Arguments:
            X(ndarray): An n by d numpy array representing n
                        data points, each with d dimensions.

        Returns:
            An n by n_components matrix representing the n data points,
            each reduced to n_components principal components.
        """

        raise NotImplementedError

    def fit_transform(self, X):
        return self.fit(X).transform(X)

#-------------------------------------------------------------------------------
# Part 2.1.1 - K-Means from scratch
#-------------------------------------------------------------------------------

class KMeans:
    def __init__(self, n_clusters, n_iter):
        """Initializes the KMeans model.

        Arguments:
            n_clusters(int): The number of clusters to learn.
            n_iter(int): The number of times to run the k-means algorithm.
        """

        self.n_clusters = n_clusters
        self.n_iter = n_iter

    def fit(self, X):
        """Learns the cluster centers self.cluster_centers_.

        Arguments:
            X(ndarray): An n by d numpy array representing n
                        data points, each with d dimensions.

        Returns:
            The fitted KMeans model.
        """

        raise NotImplementedError

    def predict(self, X):
        """Predicts which clusters new data points belong to.

        Arguments:
            X(ndarray): An n by d numpy array representing n
                        data points, each with d dimensions.

        Returns:
            A length n numpy vector containing the predicted
            cluster for each data point.
        """

        raise NotImplementedError

#-------------------------------------------------------------------------------
# Part 3.1 - Image reconstruction with autoencoders
#-------------------------------------------------------------------------------

def build_autoencoder(original_dim, encoding_dim):
    """Builds an autoencoder model.

    Layers:
        - Input, shape = (original_dim,)
        - Dense, encoding_dim neurons, relu activation
        - Dense, original_dim neurons, sigmoid activation

    Arguments:
        original_dim(int): The dimensionality of the original input
                           image (after flattening).
        encoding_dim(int): The dimensionality of the hidden layer
                           (encoding layer) of the autoencoder.

    Returns:
        A keras Model containing the autoencoder.
    """

    raise NotImplementedError

#-------------------------------------------------------------------------------
# Part 3.3 - Image generation with variational autoencoders
#-------------------------------------------------------------------------------

def build_vae(batch_size, original_dim, intermediate_dim, latent_dim, epsilon_std):
    """Builds a variational autoencoder model.

    Layers:
        - Input, shape = (original_dim,)
        - Dense, intermediate_dim neurons, relu activation
        - Dense, latent_dim neurons, linear activation (z_mean)
        - Dense, latent_dim neurons, linear activation (z_log_var)
        - Lambda, sampling function
        - Dense, intermediate_dim neurons, relu activation
        - Dense, original_dim neurons, sigmoid activation

    Arguments:
        batch_size(int): The number of inputs in each batch.
        original_dim(int): The dimensionality of the original input
                           image (after flattening).
        intermediate_dim(int): The dimensionality of the hidden layer
                               of the autoencoder.
        latent_dim(int): The dimensionality of the latent representation
                         of the input.
        epsilon_std(float): The standard deviation to use when sampling.

    Returns:
        A tuple with:
            - A keras Model containing the variational autoencoder
            - A keras Model containing the encoder portion of the vae
            - A keras Model containing the generator portion of the vae
            - A function which computes the vae loss
    """

    # Build encoder components
    x = Input(shape=(original_dim,))
    h = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    # Define sampling
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # Build decoder components
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    # Build vae
    vae = Model(x, x_decoded_mean)

    # Build encoder
    encoder = Model(x, z_mean)

    # Build generator
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)

    # Define vae loss
    def vae_loss(x, x_decoded_mean):
        xent_loss = original_dim * binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    return vae, encoder, generator, vae_loss
