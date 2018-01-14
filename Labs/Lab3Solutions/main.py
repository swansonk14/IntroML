# Partially based on http://scikit-learn.org/stable/auto_examples/svm/plot_svm_margin.html

import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics, svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import lab3
import utils

#-------------------------------------------------------------------------------
# Generating and Loading Data
#-------------------------------------------------------------------------------

# Create 100 separable data points
np.random.seed(0)
X = np.r_[np.random.randn(50, 2) - [2, 2], np.random.randn(50, 2) + [2, 2]]
y = np.array([-1] * 50 + [1] * 50)
assert len(X) == len(y)

# Plot the data
utils.plot_data(X, y)

# Load reviews data
train_data = utils.load_reviews_data('../../Data/reviews_train.csv')
val_data = utils.load_reviews_data('../../Data/reviews_val.csv')

train_texts, train_labels = zip(*((sample['text'], sample['sentiment']) for sample in train_data))
val_texts, val_labels = zip(*((sample['text'], sample['sentiment']) for sample in val_data))

#-------------------------------------------------------------------------------
# Part 1 - Pegasos Algorithm
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Part 1.1 - Implementation
#-------------------------------------------------------------------------------

theta, theta_0 = lab3.pegasos(X, y, T=5, eta=0.01, lam=0.01)
utils.plot_linear_classifier(X, y, theta, theta_0)
plt.show()

#-------------------------------------------------------------------------------
# Part 1.2 - Modifying hyperparameters
#-------------------------------------------------------------------------------

# Modify T
Ts = [1, 5, 10, 15, 20, 25]
for index, T in enumerate(Ts):
    theta, theta_0 = lab3.pegasos(X, y, T=T, eta=0.01, lam=0.01)
    subplot = int(str(int(math.ceil(math.sqrt(len(Ts))))) * 2 + str(index + 1))
    utils.plot_linear_classifier(X, y, theta, theta_0, title='T = {}'.format(T), subplot=subplot)
plt.show()

# Modify eta
etas = [10, 1, 0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6]
for index, eta in enumerate(etas):
    theta, theta_0 = lab3.pegasos(X, y, T=20, eta=eta, lam=0.01)
    subplot = int(str(int(math.ceil(math.sqrt(len(etas))))) * 2 + str(index + 1))
    utils.plot_linear_classifier(X, y, theta, theta_0, title='eta = {}'.format(eta), subplot=subplot)
plt.show()

# Modify lambda
lams = [100, 10, 1, 0.1, 0.01, 0.001]
for index, lam in enumerate(lams):
    theta, theta_0 = lab3.pegasos(X, y, T=20, eta=0.01, lam=lam)
    subplot = int(str(int(math.ceil(math.sqrt(len(lams))))) * 2 + str(index + 1))
    utils.plot_linear_classifier(X, y, theta, theta_0, title='lambda = {}'.format(lam), subplot=subplot)
plt.show()

#-------------------------------------------------------------------------------
# Part 2 - SVM
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Part 2.1 - SVM for generated data
#-------------------------------------------------------------------------------

clf = svm.SVC(kernel='linear')
clf.fit(X, y)
utils.plot_svm(X, y, clf)
plt.show()

#-------------------------------------------------------------------------------
# Part 2.2 - SVM for sentiment analysis
#-------------------------------------------------------------------------------

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('svm', svm.SVC(kernel='linear'))
])
pipeline.fit(train_texts, train_labels)

train_predictions = pipeline.predict(train_texts)
train_accuracy = metrics.accuracy_score(train_labels, train_predictions)

val_predictions = pipeline.predict(val_texts)
val_accuracy = metrics.accuracy_score(val_labels, val_predictions)

print('SVM training accuracy = {}'.format(train_accuracy))
print('SVM validation accuracy = {}'.format(val_accuracy))

#-------------------------------------------------------------------------------
# Part 3 - Grid Search
#-------------------------------------------------------------------------------

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('svm', svm.SVC(kernel='linear'))
])
param_grid = [{
    'svm__C': [1.0, 0.1, 0.01, 0.0001],
    'svm__kernel': ['linear', 'poly', 'rbf']
}]

clf = GridSearchCV(pipeline, param_grid=param_grid)
clf.fit(train_texts, train_labels)

print()
print('SVM grid search results')
print()

for C, kernel, accuracy in zip(clf.cv_results_['param_svm__C'], clf.cv_results_['param_svm__kernel'], clf.cv_results_['mean_test_score']):
    print('C = {}, kernel = {}, accuracy = {}'.format(C, kernel, accuracy))

print()
print('Best: C = {}, kernel = {}, accuracy = {}'.format(clf.best_params_['svm__C'], clf.best_params_['svm__kernel'], clf.best_score_))
