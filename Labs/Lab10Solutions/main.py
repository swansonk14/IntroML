# Dimensionality reduction based on http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html#sphx-glr-auto-examples-manifold-plot-lle-digits-py
# K-Means from sklearn based on http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py
# Clustering text documents with K-Means from http://scikit-learn.org/stable/auto_examples/text/document_clustering.html#sphx-glr-auto-examples-text-document-clustering-py
# K-Means for image compression based on http://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html#sphx-glr-auto-examples-cluster-plot-color-quantization-py
# Autoencoder based on https://blog.keras.io/building-autoencoders-in-keras.html

from keras.callbacks import TensorBoard
import matplotlib
matplotlib.rc('font', size=20)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups, load_digits, load_sample_image
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances_argmin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.random_projection import SparseRandomProjection
from sklearn.utils import shuffle

import lab10
import utils

#-------------------------------------------------------------------------------
# Part 1 - Dimensionality Reduction
#-------------------------------------------------------------------------------

# Load 8x8 digits
digits = load_digits()
X = digits.data
y = digits.target
images = digits.images

# Plot digits
utils.plot_digits(X, 'A selection from the 64-dimensional (8x8) digits dataset')

#-------------------------------------------------------------------------------
# Part 1.1 - Random Projection
#-------------------------------------------------------------------------------

print('Computing random projection')
X_rp = SparseRandomProjection(n_components=2, random_state=42).fit_transform(X)
utils.plot_embedding(X_rp, y, images, 'Random projection of 8x8 digits')

#-------------------------------------------------------------------------------
# Part 1.2 - Principal Components Analysis (PCA)
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Part 1.2.1 - PCA from scratch
#-------------------------------------------------------------------------------

print('Computing PCA')
pca = lab10.PCA(n_components=2)
X_pca = pca.fit_transform(X)
utils.plot_explained_variance_ratio(pca.explained_variance_ratio_)
utils.plot_embedding(X_pca, y, images, 'PCA (your implementation) of 8x8 digits')

#-------------------------------------------------------------------------------
# Part 1.2.2 - PCA from sklearn
#-------------------------------------------------------------------------------

print('Computing PCA')
X_pca = PCA(n_components=2).fit_transform(X)
utils.plot_embedding(X_pca, y, images, 'PCA (sklearn) of 8x8 digits')

#-------------------------------------------------------------------------------
# Part 1.3 - t-SNE
#-------------------------------------------------------------------------------

print('Computing t-SNE')
X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X)
utils.plot_embedding(X_tsne, y, images, 't-SNE of 8x8 digits')

#-------------------------------------------------------------------------------
# Part 2 - Clustering
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Part 2.1 - K-Means Algorithm
#-------------------------------------------------------------------------------

# Load 8x8 digits
digits = load_digits()
X = digits.data
y = digits.target
images = digits.images

#-------------------------------------------------------------------------------
# Part 2.1.1 - K-Means from scratch
#-------------------------------------------------------------------------------

print('Computing t-SNE')
X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X)
print('Computing K-Means')
kmeans = lab10.KMeans(n_clusters=10, n_iter=10).fit(X_tsne)
utils.plot_kmeans(X_tsne, y, kmeans, 'K-Means (your implementation) on t-SNE reduced 8x8 digits')

#-------------------------------------------------------------------------------
# Part 2.1.2 - K-Means from sklearn
#-------------------------------------------------------------------------------

print('Computing t-SNE')
X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X)
print('Computing K-Means')
kmeans = KMeans(n_clusters=10, random_state=42).fit(X_tsne)
utils.plot_kmeans(X_tsne, y, kmeans, 'K-Means (sklearn) on t-SNE reduced 8x8 digits')

#-------------------------------------------------------------------------------
# Part 2.2 - Clustering text documents with K-Means
#-------------------------------------------------------------------------------

# Load news dataset
categories = ['rec.sport.baseball', 'rec.autos', 'comp.windows.x', 'sci.space']
dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)

# Extract features using latent semantic analysis
print('Computing LSA')
tfidf = TfidfVectorizer(max_df=0.5, max_features=10000,
                        min_df=2, stop_words='english')
svd = TruncatedSVD(n_components=10)
norm = Normalizer(copy=False)
lsa = make_pipeline(tfidf, svd, norm)

X = lsa.fit_transform(dataset.data)
y = dataset.target

# Run K-Means
print('Computing K-Means')
n_clusters = len(np.unique(y))
kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)

# Determine top terms per cluster
original_space_centroids = svd.inverse_transform(kmeans.cluster_centers_)
ordered_centroids = original_space_centroids.argsort()[:, ::-1]
terms = tfidf.get_feature_names()

for i in range(n_clusters):
    top_terms = [terms[index] for index in ordered_centroids[i, :10]]
    print('Cluster {}: {}'.format(i, ', '.join(top_terms)))

# Cluster 0: window, com, mit, server, windows, motif, uk, use, host, thanks
# Cluster 1: game, team, year, baseball, games, com, cs, players, article, runs
# Cluster 2: space, nasa, gov, com, shuttle, access, henry, alaska, like, digex
# Cluster 3: car, com, cars, article, like, just, engine, oil, don, new

#-------------------------------------------------------------------------------
# Part 2.3 - Image compression with K-Means
#-------------------------------------------------------------------------------

n_colors = 64

# Load image
china = load_sample_image('china.jpg')
china = np.array(china, dtype=np.float64) / 255

width, height, depth = china.shape
china_array = np.reshape(china, (width * height, depth))
china_sample = shuffle(china_array)[:1000]

# Run K-Means and predict colors
kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(china_sample)
color_labels_kmeans = kmeans.predict(china_array)
china_array_kmeans = kmeans.cluster_centers_[color_labels_kmeans]
china_kmeans = np.reshape(china_array_kmeans, (width, height, depth))

# Predict random colors
random = shuffle(china_array, random_state=42)[:n_colors]
color_labels_random = pairwise_distances_argmin(random, china_array, axis=0)
china_array_random = random[color_labels_random]
china_random = np.reshape(china_array_random, (width, height, depth))

# Display results
ax_original = plt.subplot(221)
ax_original.imshow(china)
ax_original.set_title('Original image (96,615 colors)')

ax_kmeans = plt.subplot(223)
ax_kmeans.imshow(china_kmeans)
ax_kmeans.set_title('Quantized image (64 colors, K-Means)')

ax_random = plt.subplot(224)
ax_random.imshow(china_random)
ax_random.set_title('Quantized image (64 colors, random)')

plt.show()

#-------------------------------------------------------------------------------
# Part 3 - Autoencoders
#-------------------------------------------------------------------------------

# Load data
x_train, y_train, x_test, y_test = utils.load_mnist()

# Load noisy data
x_train_noisy, _, x_test_noisy, _ = utils.load_mnist(noisy=True)

#-------------------------------------------------------------------------------
# Part 3.1 - Image reconstruction with autoencoders
#-------------------------------------------------------------------------------

# Parameters
batch_size = 256
epochs = 50
original_dim = 784
encoding_dim = 32

# Build autoencoder
autoencoder = lab10.build_autoencoder(original_dim, encoding_dim)

# Compile autoencoder with loss function
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# Print autoencoder
autoencoder.summary()

# Train autoencoder
autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

# Use autoencoder to reconstruct test images
reconstructed_imgs = autoencoder.predict(x_test)

# Plot reconstructions
utils.plot_reconstructions(x_test, reconstructed_imgs, n=10)

#-------------------------------------------------------------------------------
# Part 3.2 - Image denoising with autoencoders
#-------------------------------------------------------------------------------

# Parameters
batch_size = 256
epochs = 50
original_dim = 784
encoding_dim = 32

# Build autoencoder
autoencoder = lab10.build_autoencoder(original_dim, encoding_dim)

# Compile autoencoder with loss function
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# Print autoencoder
autoencoder.summary()

# Train autoencoder
autoencoder.fit(x_train_noisy, x_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_test_noisy, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/denoising')])

# Use autoencoder to denoise test images
denoised_imgs = autoencoder.predict(x_test_noisy)

# Plot denoised images
utils.plot_reconstructions(x_test_noisy, denoised_imgs, n=10)

#-------------------------------------------------------------------------------
# Part 3.3 - Image generation with variational autoencoders
#-------------------------------------------------------------------------------

# Parameters
batch_size = 256
epochs = 50
original_dim = 784
intermediate_dim = 256
latent_dim = 2
epsilon_std = 1.0

# Build variational autoencoder (vae)
vae, encoder, generator, vae_loss = lab10.build_vae(batch_size,
                                                    original_dim,
                                                    intermediate_dim,
                                                    latent_dim,
                                                    epsilon_std)

# Compile vae with loss function
vae.compile(optimizer='rmsprop', loss=vae_loss)

# Print vae
vae.summary()

# Train vae
vae.fit(x_train, x_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_test, x_test),
        callbacks=[TensorBoard(log_dir='/tmp/vae')])

# Use vae to reconstruct test images
reconstructed_imgs = vae.predict(x_test)

# Plot reconstructions
utils.plot_reconstructions(x_test, reconstructed_imgs, n=10)

# Plot encoding of vae
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test, cmap=plt.cm.rainbow, edgecolors='k')
plt.colorbar()
plt.title('Dimensionality reduction with variational autoencoder')
plt.show()

# Generate digits with vae
utils.plot_generated_digits_vae(generator, epsilon_std)
