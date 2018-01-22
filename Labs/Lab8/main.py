# Additional datasets partially based on http://scikit-learn.org/stable/auto_examples/datasets/plot_random_dataset.html#sphx-glr-auto-examples-datasets-plot-random-dataset-py

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_classification, make_gaussian_quantiles, make_moons
from tqdm import tqdm

import lab8
import utils

#-------------------------------------------------------------------------------
# Generating Data
#-------------------------------------------------------------------------------

# Generate Data
X, y = make_moons(200, noise=0.2, random_state=0)

# Plot data
ax = plt.gca()
utils.plot_data(ax, X, y, title='Original Moons Data')
plt.show()

#-------------------------------------------------------------------------------
# Part 4 - Testing our Neural Network
#-------------------------------------------------------------------------------

# nn = lab8.NN(n_input=X.shape[1], n_hidden=50, n_output=2)
# nn.train(X, y, num_epochs=100, verbose=True)

# ax = plt.gca()
# utils.plot_decision_boundary(ax, X, y, nn, title='Neural Network Moons')
# plt.show()

#-------------------------------------------------------------------------------
# Part 5 - Experimenting with the Number of Neurons
#-------------------------------------------------------------------------------

# hidden_layer_dimensions = [1, 2, 3, 4, 5, 6, 10, 20, 50]
# for i, n_hidden in tqdm(enumerate(hidden_layer_dimensions), total=len(hidden_layer_dimensions)):
#     nn = lab8.NN(n_input=X.shape[1], n_hidden=n_hidden, n_output=2)
#     nn.train(X, y, num_epochs=2000)

#     ax = plt.subplot(3, 3, i + 1)
#     utils.plot_decision_boundary(ax, X, y, nn, title='Neural Network with {} Hidden Units'.format(n_hidden))
# plt.show()

#-------------------------------------------------------------------------------
# Part 6 - Neural Networks with Keras
#-------------------------------------------------------------------------------

# # Modify label dimension for keras
# n = X.shape[0]
# y_2d = np.zeros((n, 2))
# y_2d[range(n), y] = 1

# # Build network
# model = lab8.keras_nn(n_input=X.shape[1], n_hidden=50, n_output=2)
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Train network
# history = model.fit(X, y_2d, epochs=500, batch_size=10)

# # Plot loss and accuracy
# utils.plot_loss_and_accuracy(history)
# plt.show()

# # Plot decision boundary
# ax = plt.gca()
# utils.plot_decision_boundary(ax, X, y, model, title='Keras Neural Network Moons', multiclass=True)
# plt.show()

#-------------------------------------------------------------------------------
# Part 7 - Clustering Dataset with Keras
#-------------------------------------------------------------------------------

# # Generate data
# X_cluster, y_cluster = make_blobs(n_samples=200, centers=[[0,4], [-4,0], [4,0], [0,-4], [0,0]], random_state=2)

# # Plot data
# ax = plt.gca()
# utils.plot_data(ax, X_cluster, y_cluster, title='Original Clusters Data')
# plt.show()

# # Modify label dimension for keras
# n = X_cluster.shape[0]
# y_2d_cluster = np.zeros((n, 5))
# y_2d_cluster[range(n), y_cluster] = 1

# # Build and train network
# model = lab8.keras_nn(n_input=X_cluster.shape[1], n_hidden=50, n_output=5)
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X_cluster, y_2d_cluster, epochs=100, batch_size=10)

# # Plot decision boundary
# ax = plt.gca()
# utils.plot_decision_boundary(ax, X_cluster, y_cluster, model, title='Keras Neural Network Clusters', multiclass=True)
# plt.show()

#-------------------------------------------------------------------------------
# Part 8 - Additional Datasets with Keras
#-------------------------------------------------------------------------------

# # Generate data
# X1, y1 = make_classification(n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1, random_state=2)
# X2, y2 = make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=2)
# X3, y3 = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=2)
# X4, y4 = make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, n_classes=3, random_state=2)
# X5, y5 = make_blobs(n_features=2, centers=3, random_state=2)
# X6, y6 = make_gaussian_quantiles(n_features=2, n_classes=3, random_state=2)

# Xs = [X1, X2, X3, X4, X5, X6]
# ys = [y1, y2, y3, y4, y5, y6]
# n_classes = [2]*3 + [3]*3
# titles = ['One informative feature, one cluster per class',
#           'Two informative features, one cluster per class',
#           'Two informative features, two clusters per class',
#           'Multi-class, two informative features, one cluster',
#           'Three blobs',
#           'Gaussian divided into three quantiles']

# # Plot data
# for i in range(len(Xs)):
#     ax = plt.subplot(3, 2, i+1)
#     utils.plot_data(ax, Xs[i], ys[i], title=titles[i])
# plt.show()

# # Train network and plot decision boundaries
# for i in range(len(Xs)):
#     # Modify label dimension for keras
#     n = Xs[i].shape[0]
#     y_2d = np.zeros((n, n_classes[i]))
#     y_2d[range(n), ys[i]] = 1

#     # Build and train network
#     model = lab8.keras_nn(n_input=Xs[i].shape[1], n_hidden=50, n_output=n_classes[i])
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.fit(Xs[i], y_2d, epochs=100, batch_size=10)

#     # Plot decision boundary
#     ax = plt.subplot(3, 2, i+1)
#     utils.plot_decision_boundary(ax, Xs[i], ys[i], model, title=titles[i], multiclass=True)
# plt.show()

#-------------------------------------------------------------------------------
# Part 9 - Free Experimentation
#-------------------------------------------------------------------------------

# YOUR CODE HERE
