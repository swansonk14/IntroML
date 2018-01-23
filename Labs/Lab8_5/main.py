import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import lab8_5
import utils

#-------------------------------------------------------------------------------
# Part 1 - Pythagorean Function
#-------------------------------------------------------------------------------

# # Define pythagorean function
# def pythagorean(x, y):
#     return np.sqrt(np.square(x) + np.square(y))

# # Generate pythagorean data
# X, y = utils.generate_3d_data(low=-100,
#                               high=100,
#                               n=10000,
#                               func=pythagorean)

#-------------------------------------------------------------------------------
# Part 1.1 - Learning the Pythagorean Function
#-------------------------------------------------------------------------------

# # Build single layer regression model
# model = lab8_5.build_single_layer_regression_model(n_hidden=50)
# model.summary()
# model.compile(loss='mean_squared_error', optimizer='adam')

# # Plot model predictions for different numbers of epochs
# for i, epochs in enumerate([10, 20, 100, 500]):
#     model.fit(X, y, epochs=epochs, batch_size=128)

#     ax = plt.subplot(2, 2, i+1, projection='3d')
#     ax.set_title('Number of epochs = {}'.format(epochs))
#     utils.plot_predictions_3d(ax_data=ax,
#                               ax_pred=ax,
#                               low=-100,
#                               high=100,
#                               func=pythagorean,
#                               model=model)
# plt.show()

#-------------------------------------------------------------------------------
# Part 1.2 - Generalizing the Pythagorean Function
#-------------------------------------------------------------------------------

# # Build single layer regression model
# model = lab8_5.build_single_layer_regression_model(n_hidden=50)
# model.summary()
# model.compile(loss='mean_squared_error', optimizer='adam')

# model.fit(X, y, epochs=500, batch_size=128)

# # Plot model predictions over a range it didn't train on
# ax_data = plt.subplot(1, 2, 1, projection='3d')
# ax_data.set_title('Pythagorean function')
# ax_pred = plt.subplot(1, 2, 2, projection='3d')
# ax_pred.set_title('Model prediction')
# utils.plot_predictions_3d(ax_data=ax_data,
#                           ax_pred=ax_pred,
#                           low=-200,
#                           high=200,
#                           func=pythagorean,
#                           model=model)
# plt.show()

#-------------------------------------------------------------------------------
# Part 2 - Ripple Function
#-------------------------------------------------------------------------------

# # Define ripple function
# def ripple(x, y):
#     return np.sin(10 * (np.square(x) + np.square(y)))

# # Generate ripple data
# X, y = utils.generate_3d_data(low=-1,
#                               high=1,
#                               n=10000,
#                               func=ripple)

#-------------------------------------------------------------------------------
# Part 2.1 - Limits of the Universal Approximation Theorem
#-------------------------------------------------------------------------------

# # Parameters
# n_hidden = 1000
# epochs = 200

# # Build single layer regression model
# model = lab8_5.build_single_layer_regression_model(n_hidden=n_hidden)
# model.summary()
# model.compile(loss='mean_squared_error', optimizer='adam')

# model.fit(X, y, epochs=epochs, batch_size=128)

# # Plot model predictions
# ax_data = plt.subplot(1, 2, 1, projection='3d')
# ax_data.set_title('Ripple function')
# ax_pred = plt.subplot(1, 2, 2, projection='3d')
# ax_pred.set_title('Model prediction')
# utils.plot_predictions_3d(ax_data=ax_data,
#                           ax_pred=ax_pred,
#                           low=-1,
#                           high=1,
#                           func=ripple,
#                           model=model)
# plt.show()

#-------------------------------------------------------------------------------
# Part 2.2 - Power of Deep Networks
#-------------------------------------------------------------------------------

# # Parameters
# n_hidden = 50
# n_layers = 3
# epochs = 200

# # Build deep regression model
# model = lab8_5.build_deep_regression_model(n_hidden=n_hidden,
#                                            n_layers=n_layers)
# model.summary()
# model.compile(loss='mean_squared_error', optimizer='adam')

# model.fit(X, y, epochs=epochs, batch_size=128)

# # Plot model predictions
# ax_data = plt.subplot(1, 2, 1, projection='3d')
# ax_data.set_title('Ripple function')
# ax_pred = plt.subplot(1, 2, 2, projection='3d')
# ax_pred.set_title('Model prediction')
# utils.plot_predictions_3d(ax_data=ax_data,
#                           ax_pred=ax_pred,
#                           low=-1,
#                           high=1,
#                           func=ripple,
#                           model=model)
# plt.show()

#-------------------------------------------------------------------------------
# Part 3 - Custom Functions
#-------------------------------------------------------------------------------

# # Parameters
# low = -10
# high = 10

# # Define custom function
# def custom(x, y):
#     raise NotImplementedError

# # Generate custom data
# X, y = utils.generate_3d_data(low=low,
#                               high=high,
#                               n=10000,
#                               func=custom)

#-------------------------------------------------------------------------------
# Part 3.1 - Single Layer Neural Network
#-------------------------------------------------------------------------------

# # Parameters
# n_hidden = 50
# epochs = 500

# # Build single layer regression model
# model = lab8_5.build_single_layer_regression_model(n_hidden=n_hidden)
# model.summary()
# model.compile(loss='mean_squared_error', optimizer='adam')

# model.fit(X, y, epochs=epochs, batch_size=128)

# # Plot model predictions
# ax_data = plt.subplot(1, 2, 1, projection='3d')
# ax_data.set_title('Custom function')
# ax_pred = plt.subplot(1, 2, 2, projection='3d')
# ax_pred.set_title('Model prediction')
# utils.plot_predictions_3d(ax_data=ax_data,
#                           ax_pred=ax_pred,
#                           low=low,
#                           high=high,
#                           func=custom,
#                           model=model)
# plt.show()

#-------------------------------------------------------------------------------
# Part 3.2 - Deep Neural Network
#-------------------------------------------------------------------------------

# # Parameters
# n_hidden = 50
# n_layers = 3
# epochs = 500

# # Build deep regression model
# model = lab8_5.build_deep_regression_model(n_hidden=n_hidden,
#                                            n_layers=n_layers)
# model.summary()
# model.compile(loss='mean_squared_error', optimizer='adam')

# model.fit(X, y, epochs=epochs, batch_size=128)

# # Plot model predictions
# ax_data = plt.subplot(1, 2, 1, projection='3d')
# ax_data.set_title('Custom function')
# ax_pred = plt.subplot(1, 2, 2, projection='3d')
# ax_pred.set_title('Model prediction')
# utils.plot_predictions_3d(ax_data=ax_data,
#                           ax_pred=ax_pred,
#                           low=low,
#                           high=high,
#                           func=custom,
#                           model=model)
# plt.show()
