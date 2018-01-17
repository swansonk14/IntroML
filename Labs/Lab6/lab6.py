import numpy as np

#-------------------------------------------------------------------------------
# Part 1 - Naive Recommendation Algorithm
#-------------------------------------------------------------------------------

def predict_ratings_naive(Y):
    """Uses a naive algorithm for predicting movie ratings.

    If the user/movie rating is known, predict the known rating.
    Otherwise predict the average rating for that movie.
    If there are no ratings for the movie, predict the average rating
    of all movies.

    Arguments:
        Y(ndarray): A matrix of ratings where each row represents a
                    user and each column represents a movie.
                    -1 represents and unknown rating.

    Returns:
        A matrix with the predicted ratings for all users/movies.
    """

    raise NotImplementedError

#-------------------------------------------------------------------------------
# Part 2 - Nearest-Neighbor Prediction
#-------------------------------------------------------------------------------

def predict_ratings_nearest_neighbor(Y, k=10):
    """Uses user similarity to make a nearest neighbor prediction.

    First computes the similarity (correlation) between users
    based on how users rate movies that both have seen.
    Then to make predictions for a user, the algorithm finds
    the k most similar users who have seen the movie and averages
    their ratings. However, rather than directly averaging their
    ratings, the algorithm instead uses the mean rating given by
    this user and adds the difference between the ratings for this
    movie and the mean ratings for each of the other users, weighted
    by similarity.

    Arguments:
        Y(ndarray): A matrix of ratings where each row represents a
                    user and each column represents a movie.
                    -1 represents and unknown rating.
        k(int): The number of nearest neighbors to use.

    Returns:
        A matrix with the predicted ratings for all users/movies.
    """

    raise NotImplementedError

#-------------------------------------------------------------------------------
# Part 3 - Matrix Factorization
#-------------------------------------------------------------------------------

# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.173.2797&rep=rep1&type=pdf
def predict_ratings_matrix_factorization(R, nf=20, lam=0.05, T=20):
    """Uses low-rank matrix factorization to predict movie ratings.

    Arguments:
        R(ndarray): A matrix of ratings where each row represents a
                    user and each column represents a movie.
                    -1 represents and unknown rating. Equivalent to Y.
        nf(int): The number of features for each user/movie. Equivalent
                 to the rank k of the matrices U and M.
        lam(float): The lambda value.
        T(int): The number of times to perform the alternating minimization.

    Returns:
        A matrix with the predicted ratings for all users/movies.
    """

    raise NotImplementedError
