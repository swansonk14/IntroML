from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import utils

#-------------------------------------------------------------------------------
# Part 1 - Data Loading
#-------------------------------------------------------------------------------

data = load_breast_cancer()
X, y = data['data'], data['target']
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

#-------------------------------------------------------------------------------
# Part 2 - Decision Tree Classifier
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Part 2.1 - Building and Training a Decision Tree Classifier
#-------------------------------------------------------------------------------

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

#-------------------------------------------------------------------------------
# Part 2.2 - Evaluating the Decision Tree Classifier
#-------------------------------------------------------------------------------

print('Decision tree accuracy on cancer dataset:')

# Evaluate train accuracy
y_train_pred = clf.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print('train accuracy = {:.4f}'.format(train_accuracy)) # 1.0000

# Evaluate test accuracy
y_test_pred = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('test accuracy = {:.4f}'.format(test_accuracy)) # 0.9474

print()

#-------------------------------------------------------------------------------
# Part 2.3 - Interpreting the Decision Tree Classifier
#-------------------------------------------------------------------------------

# Print most important features for random forest
sorted_features = utils.sort_features(data['feature_names'],
                                      clf.feature_importances_)

print('Decision tree features (sorted by importance):')
print('\n'.join('{}: {:.4f}'.format(feature, importance) for feature, importance in sorted_features))
print()

"""
mean concave points: 0.6914
worst concave points: 0.0657
mean texture: 0.0585
worst radius: 0.0523
worst perimeter: 0.0515
etc.
"""

# Display decision tree
utils.display_decision_tree(clf,
                            data['feature_names'],
                            data['target_names'])

#-------------------------------------------------------------------------------
# Part 3 - Random Forest Classifier
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Part 3.1 - Building and Training a Random Forest Classifier
#-------------------------------------------------------------------------------

# Train random forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

#-------------------------------------------------------------------------------
# Part 3.2 - Evaluating the Random Forest Classifier
#-------------------------------------------------------------------------------

print('Random forest accuracy on cancer dataset:')

# Evaluate train accuracy
y_train_pred = clf.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print('train accuracy = {:.4f}'.format(train_accuracy)) # 0.9978

# Evaluate test accuracy
y_test_pred = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('test accuracy = {:.4f}'.format(test_accuracy)) # 0.9561

print()

#-------------------------------------------------------------------------------
# Part 3.3 - Interpreting the Random Forest Classifier
#-------------------------------------------------------------------------------

# Display decision trees
for index, tree in enumerate(clf.estimators_):
    # Print most important features for decision tree
    sorted_features = utils.sort_features(data['feature_names'],
                                          tree.feature_importances_)

    print('tree {} features (sorted by importance):'.format(index + 1))
    print('\n'.join('{}: {:.4f}'.format(feature, importance) for feature, importance in sorted_features))
    print()

    # Display decision tree
    utils.display_decision_tree(tree,
                                data['feature_names'],
                                data['target_names'])

# Print most important features for random forest
sorted_features = utils.sort_features(data['feature_names'],
                                      clf.feature_importances_)

print('Random forest features (sorted by importance):')
print('\n'.join('{}: {:.4f}'.format(feature, importance) for feature, importance in sorted_features))

"""
worst concave points: 0.2374
mean concave points: 0.1100
worst area: 0.0971
worst radius: 0.0834
mean radius: 0.0707
etc.
"""
