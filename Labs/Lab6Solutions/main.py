import lab6
import utils

#-------------------------------------------------------------------------------
# Data Loading
#-------------------------------------------------------------------------------

Y = utils.load_movies_data('../../Data/movie_ratings.csv')

Y_train, Y_test = utils.train_test_split(Y)

#-------------------------------------------------------------------------------
# Part 1 - Naive Recommendation Algorithm
#-------------------------------------------------------------------------------

print('Naive Algorithm')

X_naive = lab6.predict_ratings_naive(Y_train)

rmse_train = utils.root_mean_squared_error(Y_train, X_naive) # 0.0000
print('train rmse = {:.4f}'.format(rmse_train))

rmse_test = utils.root_mean_squared_error(Y_test, X_naive) # 1.0018
print('test rmse = {:.4f}'.format(rmse_test))

print()

#-------------------------------------------------------------------------------
# Part 2 - Nearest-Neighbor Prediction
#-------------------------------------------------------------------------------

print('Nearest-Neighbor Prediction')

X_nn = lab6.predict_ratings_nearest_neighbor(Y_train)

rmse_train = utils.root_mean_squared_error(Y_train, X_nn) # 0.7857
print('train rmse = {:.4f}'.format(rmse_train))

rmse_test = utils.root_mean_squared_error(Y_test, X_nn) # 0.9393
print('test rmse = {:.4f}'.format(rmse_test))

print()

#-------------------------------------------------------------------------------
# Part 3 - Matrix Factorization
#-------------------------------------------------------------------------------

print('Matrix Factorization')

X_matrix = lab6.predict_ratings_matrix_factorization(Y_train)

rmse_train = utils.root_mean_squared_error(Y_train, X_matrix) # 0.8047
print('train rmse = {:.4f}'.format(rmse_train))

rmse_test = utils.root_mean_squared_error(Y_test, X_matrix) # 1.1108
print('test rmse = {:.4f}'.format(rmse_test))
