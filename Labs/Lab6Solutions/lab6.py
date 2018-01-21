from itertools import product

import numpy as np
from tqdm import tqdm, trange

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

    num_users, num_movies = Y.shape

    X = np.zeros((num_users, num_movies))

    average_rating = np.mean(Y[np.where(Y != -1)])

    for i in trange(num_movies):
        # Determine known ratings for movie i
        movie_ratings = Y[:,i]
        known_movie_ratings = movie_ratings[np.where(movie_ratings != -1)]

        if len(known_movie_ratings) != 0:
            average_movie_rating = np.mean(known_movie_ratings)
        else:
            average_movie_rating = -1

        # Assign ratings for this movie for each user
        for a in range(num_users):
            # If user has rated the movie, just use that rating
            if Y[a,i] != -1:
                X[a,i] = Y[a,i]

            # Otherwise use the average movie rating if it exists
            elif average_movie_rating != -1:
                X[a,i] = average_movie_rating

            # Otherwise default to average rating overall
            else:
                X[a,i] = average_rating

    return X

#-------------------------------------------------------------------------------
# Part 2 - Nearest-Neighbor Prediction
#-------------------------------------------------------------------------------

def compute_user_similarity(Y, a, b):
    """Computes the similarity between two users.

    The similarity (correlation) between two users
    is based on how often the two users rate movies
    similarly.

    Default similarity of 0 if no movies in common.

    Arguments:
        Y(ndarray): A matrix of ratings where each row represents a
                    user and each column represents a movie.
                    -1 represents and unknown rating.
        a(int): The index of the first user.
        b(int): The index of the second user.

    Returns:
        A float in the range [-1,1] indicating the
        similarity of the users.
    """

    # Get the movie ratings of users a and b
    movie_ratings_a = Y[a]
    movie_ratings_b = Y[b]

    # Determine which movies both users have seen
    movies_in_common = np.where(np.logical_and(movie_ratings_a != -1,
                                               movie_ratings_b != -1))[0]

    # If no movies in common, default similarity is 0
    if len(movies_in_common) == 0:
        return 0

    # Get the movie ratings for movies in common of a and b
    movie_ratings_common_a = movie_ratings_a[movies_in_common]
    movie_ratings_common_b = movie_ratings_b[movies_in_common]

    # Compute mean ratings for the movies in common
    mean_movie_rating_common_a = np.mean(movie_ratings_a[movies_in_common])
    mean_movie_rating_common_b = np.mean(movie_ratings_b[movies_in_common])

    # Compute covariance and variances
    cov = np.dot(movie_ratings_common_a - mean_movie_rating_common_a,
                 movie_ratings_common_b - mean_movie_rating_common_b)
    var_a = np.sqrt(np.sum((movie_ratings_common_a - mean_movie_rating_common_a)**2))
    var_b = np.sqrt(np.sum((movie_ratings_common_b - mean_movie_rating_common_b)**2))

    # Return 0 if one of the variances is 0 because then a and b are uncorrelated
    if var_a == 0 or var_b == 0:
        return 0

    # Compute similarity (correlation)
    similarity = cov / (var_a * var_b)

    return similarity

def compute_user_similarity_matrix(Y):
    """Computes a matrix of similarities between users.

    The similarity (correlation) between two users
    is based on how often the two users rate movies
    similarly.

    Default similarity of 0 if no movies in common.

    Arguments:
        Y(ndarray): A matrix of ratings where each row represents a
                    user and each column represents a movie.
                    -1 represents and unknown rating.

    Returns:
        A num_users by num_users matrix where entry (a,b)
        contains the similarity between users a and b.
    """

    num_users = Y.shape[0]

    S = np.ones((num_users, num_users))

    for a in trange(num_users):
        for b in range(a+1, num_users):
            S[a,b] = S[b,a] = compute_user_similarity(Y, a, b)

    return S

def compute_knn(Y, S, k, a, i):
    """Determines the k users most similar to user a who have seen movie i.

    If multiple users have equal similarities, users
    are randomly selected.

    If fewer than k users have seen movie i, then all
    users who have seen movie i are returned.

    Arguments:
        Y(ndarray): A matrix of ratings where each row represents a
                    user and each column represents a movie.
                    -1 represents and unknown rating.
        S(ndarray): A num_users by num_users matrix where
                    entry (a,b) contains the similarity
                    between users a and b.
        k(int): The number of similar users to find for
                each user.
        a(int): The user index.
        i(int): The movie index.

    Returns:
        A numpy vector with the indices of the k users
        who have seen movie i who are most similar
        to user a.
    """

    # Determine which users (who are not a) who have seen movie i
    users_seen_i = np.where(Y[:,i] != -1)[0]
    users_seen_i = np.delete(users_seen_i, np.where(users_seen_i == a))# np.setdiff1d(users_seen_i, [a])

    # Get the top k users ordered by similarity
    if len(users_seen_i) > k:
        top_k_indices = np.argpartition(S[a,users_seen_i], -k)[-k:]
        knn = users_seen_i[top_k_indices]
    else:
        knn = users_seen_i

    return knn

def predict_ratings_nn(Y, S, KNN):
    """Predict ratings using nearest neighbor prediction.

    Arguments:
        Y(ndarray): A matrix of ratings where each row represents a
                    user and each column represents a movie.
                    -1 represents and unknown rating.
        S(ndarray): A num_users by num_users matrix where
                    entry (a,b) contains the similarity
                    between users a and b.
        KNN(dict): A dictionary mapping user/movie tuples (a,i)
                   to a numpy array of most similar user indices.

    Returns:
        A matrix with the predicted ratings for all users/movies.
    """

    num_users, num_movies = Y.shape

    X = np.zeros(Y.shape)

    user_mean_movie_ratings = np.zeros(num_users)
    for a in range(num_users):
        user_mean_movie_ratings[a] = np.mean(Y[a][np.where(Y[a] != -1)])

    for a in trange(num_users):
        for i in range(num_movies):
            similar_users_deviation = np.dot(S[a,KNN[(a,i)]],
                                             Y[KNN[(a,i)],i] - user_mean_movie_ratings[KNN[(a,i)]])
            total_similarity = np.sum(np.abs(S[a,KNN[(a,i)]]))

            if total_similarity == 0:
                X[a,i] = user_mean_movie_ratings[a]
            else:
                X[a,i] = user_mean_movie_ratings[a] + similar_users_deviation / total_similarity

    return X

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

    num_users, num_movies = Y.shape

    # Determine similarities between users
    S = compute_user_similarity_matrix(Y)

    # Determine k most similar users to user a who have seen movie i for every user a and movie i
    KNN = {(a,i): compute_knn(Y, S, k, a, i) for a, i in tqdm(product(range(num_users),
                                                                      range(num_movies)),
                                                              total=num_users*num_movies)}
    
    # Predict ratings
    X = predict_ratings_nn(Y, S, KNN)

    return X

#-------------------------------------------------------------------------------
# Part 3 - Matrix Factorization
#-------------------------------------------------------------------------------

def optimize_U(R, M, num_users, nf, lam):
    """Optimizes the U matrix for matrix factorization.

    Arguments:
        R(ndarray): A matrix of ratings where each row represents a
                    user and each column represents a movie.
                    -1 represents and unknown rating. Equivalent to Y.
        M(ndarray): The M matrix in the matrix factorization of R.
        num_movies(int): The number of movies.
        nf(int): The number of features for each user/movie. Equivalent
                 to the rank k of the matrices U and M.
        lam(float): The lambda value.

    Returns:
        The optimized U matrix.
    """

    U = np.zeros((nf, num_users))

    for i in range(num_users):
        Ii = np.where(R[i] != -1)[0] # movies rated by user i
        MIi = M[:,Ii] # columns of M.T (movies) for which user i has given a rating
        nui = len(Ii) # number of movies rated by user i
        E = np.eye(nf) # identify matrix of size nf x nf

        Ai = np.dot(MIi, MIi.T) + lam * nui * E

        RiIi = R[i,Ii] # ith row vector of R but only with entries from columns rated by user i

        Vi = np.dot(MIi, RiIi.T)

        U[:,i] = np.dot(np.linalg.inv(Ai), Vi)

    return U

def optimize_M(R, U, num_movies, nf, lam):
    """Optimizes the M matrix for matrix factorization.

    Arguments:
        R(ndarray): A matrix of ratings where each row represents a
                    user and each column represents a movie.
                    -1 represents and unknown rating. Equivalent to Y.
        U(ndarray): The U matrix in the matrix factorization of R.
        num_movies(int): The number of movies.
        nf(int): The number of features for each user/movie. Equivalent
                 to the rank k of the matrices U and M.
        lam(float): The lambda value.

    Returns:
        The optimized M matrix.
    """

    M = np.zeros((nf, num_movies))

    for j in range(num_movies):
        Ij = np.where(R[:,j] != -1)[0] # users who rated movie j
        UIj = U[:,Ij] # columns of U.T (users) who have rated movie j
        nmj = len(Ij) # number of users who rated movie j
        E = np.eye(nf) # identify matrix of size nf x nf

        Aj = np.dot(UIj, UIj.T) + lam * nmj * E

        RIjj = R[Ij,j] # jth column vector of R but only with entries from rows which rated movie j
        RIjj = RIjj.reshape(RIjj.size,1)

        Vj = np.dot(UIj, RIjj).flatten()

        M[:,j] = np.dot(np.linalg.pinv(Aj), Vj)

    return M

# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.173.2797&rep=rep1&type=pdf
def predict_ratings_matrix_factorization(R, nf=1, lam=0.05, T=10):
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

    num_users, num_movies = R.shape

    np.random.seed(0)
    M = np.random.rand(nf, num_movies)
    M[0] = np.mean(R, axis=0)

    for _ in trange(T):
        U = optimize_U(R, M, num_users, nf, lam)
        M = optimize_M(R, U, num_movies, nf, lam)

    X = np.dot(U.T, M)

    return X
