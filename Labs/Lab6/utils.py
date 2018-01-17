import csv

import numpy as np
from sklearn.metrics import mean_squared_error

# Data: https://grouplens.org/datasets/movielens/latest/
def load_movies_data(movies_data_path):
    """Loads the movies dataset as a numpy matrix.

    Arguments:
        movies_data_path(str): Path to the movies dataset .csv file.

    Returns:
        A numpy matrix where the rows are users and the columns are movies.
        The value in entry (i,j) indicates the rating given by user i to 
        movie j. A value of -1 indicates that the user has not rated that movie.
    """

    userId_to_index = {}
    movieId_to_index = {}
    user_movie_to_rating = {}

    with open(movies_data_path) as data_file:
        for datum in csv.DictReader(data_file, delimiter=','):
            userId = int(datum['userId'])
            movieId = int(datum['movieId'])
            rating = float(datum['rating'])

            user_index = userId_to_index.setdefault(userId, len(userId_to_index))
            movie_index = movieId_to_index.setdefault(movieId, len(movieId_to_index))

            user_movie_to_rating[(user_index, movie_index)] = rating

    num_users = len(userId_to_index)
    num_movies = len(movieId_to_index)

    Y = np.zeros((num_users, num_movies)) - 1

    for user_movie, rating in user_movie_to_rating.items():
        Y[user_movie] = rating

    return Y

def train_test_split(Y):
    """Given a matrix of ratings, returns a train and test matrix of ratings.

    Randomly selects 80% of known ratings in the matrix as train
    set and the remaining 20% are the test set.
    Generates two matrices, one with only the train ratings (all
    other entries are -1) and one with only the test ratings (all
    other entires are -1).

    Arguments:
        Y(ndarray): A matrix of ratings where each row represents a
                    user and each column represents a movie.
                    -1 represents and unknown rating.

    Returns:
        A tuple of matrices. The first is a train matrix with 80% of the
        known ratings and the second is a test matrix with the remaining
        20% of the known ratings.
    """

    rated_indices = np.column_stack(np.where(Y != -1))

    num_rated = len(rated_indices)
    num_train = int(np.ceil(0.8 * num_rated))

    np.random.seed(0)
    train_indices = np.random.choice(range(num_rated), size=num_train, replace=False)
    test_indices = np.setdiff1d(range(num_rated), train_indices)

    rated_indices_train = rated_indices[train_indices]
    rated_indices_test = rated_indices[test_indices]

    Y_train = np.copy(Y)
    Y_train[(rated_indices_test[:,0], rated_indices_test[:,1])] = -1

    Y_test = np.copy(Y)
    Y_test[(rated_indices_train[:,0], rated_indices_train[:,1])] = -1

    return Y_train, Y_test

def root_mean_squared_error(Y, X):
    """Compute the root mean squared error for known entries of Y.

    Arguments:
        Y(ndarray): A matrix with the correct ratings.
        X(ndarray): A matrix with the predicted ratings.

    Returns:
        A float with the root mean squared error between predictions
        in X and known ratings in Y.
    """

    Y_known = Y[np.where(Y != -1)]
    X_known = X[np.where(Y != -1)]

    mse = mean_squared_error(Y_known, X_known)
    rmse = np.sqrt(mse)

    return rmse
