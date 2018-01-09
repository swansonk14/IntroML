import lab2
import utils

#-------------------------------------------------------------------------------
# Data Loading
#-------------------------------------------------------------------------------

train_data = utils.load_reviews_data('../../Data/reviews_train.csv')
val_data = utils.load_reviews_data('../../Data/reviews_val.csv')
test_data = utils.load_reviews_data('../../Data/reviews_test.csv')

train_texts, train_labels = zip(*((sample['text'], sample['sentiment']) for sample in train_data))
val_texts, val_labels = zip(*((sample['text'], sample['sentiment']) for sample in val_data))
test_texts, test_labels = zip(*((sample['text'], sample['sentiment']) for sample in test_data))

dictionary = lab2.bag_of_words(train_texts)

train_bow_features = lab2.extract_bow_feature_vectors(train_texts, dictionary)
val_bow_features = lab2.extract_bow_feature_vectors(val_texts, dictionary)
test_bow_features = lab2.extract_bow_feature_vectors(test_texts, dictionary)

# You may modify the following when adding additional features (Part 3c)

train_final_features = lab2.extract_final_features(train_texts, dictionary)
val_final_features = lab2.extract_final_features(val_texts, dictionary)
test_final_features = lab2.extract_final_features(test_texts, dictionary)

#-------------------------------------------------------------------------------
# Part 1 - Perceptron Algorithm
#-------------------------------------------------------------------------------

# toy_features, toy_labels = utils.load_toy_data('../../Data/toy_data.csv')

# theta, theta_0 = lab2.perceptron(toy_features, toy_labels, T=5)

# utils.plot_toy_results(toy_features, toy_labels, theta, theta_0)

#-------------------------------------------------------------------------------
# Part 2 - Classifying Reviews
#-------------------------------------------------------------------------------

# theta, theta_0 = lab2.perceptron(train_bow_features, train_labels, T=5)

# train_accuracy = lab2.accuracy(train_bow_features, train_labels, theta, theta_0)
# val_accuracy = lab2.accuracy(val_bow_features, val_labels, theta, theta_0)

# print("Training accuracy: {:.4f}".format(train_accuracy))
# print("Validation accuracy: {:.4f}".format(val_accuracy))

#-------------------------------------------------------------------------------
# Part 3 - Improving the Model
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Part 3.1 - Tuning the Hyperparameters
#-------------------------------------------------------------------------------

# Ts = [1, 5, 10, 15, 20]

# train_accs, val_accs = lab2.tune(Ts, train_bow_features, train_labels, val_bow_features, val_labels)

# utils.plot_tune_results(Ts, train_accs, val_accs)

#-------------------------------------------------------------------------------
# Best T value
#-------------------------------------------------------------------------------

T_best = 1 # You may modify this value

#-------------------------------------------------------------------------------
# Part 3.2 - Understanding the Model
#-------------------------------------------------------------------------------

# theta, theta_0 = lab2.perceptron(train_bow_features, train_labels, T_best)

# word_list = sorted(dictionary.keys(), key=lambda word: dictionary[word])
# sorted_words = utils.most_explanatory_words(theta, word_list)

# print("Top 10 most explanatory words")
# print(sorted_words[:10])

#-------------------------------------------------------------------------------
# Part 3.3 - Adding Features
#-------------------------------------------------------------------------------

# theta, theta_0 = lab2.perceptron(train_bow_features, train_labels, T_best)

# train_accuracy = lab2.accuracy(train_bow_features, train_labels, theta, theta_0)
# val_accuracy = lab2.accuracy(val_bow_features, val_labels, theta, theta_0)

# print("Bag-of-words features")
# print("Training accuracy: {:.4f}".format(train_accuracy))
# print("Validation accuracy: {:.4f}".format(val_accuracy))

# print()

# theta, theta_0 = lab2.perceptron(train_final_features, train_labels, T_best)

# train_accuracy = lab2.accuracy(train_final_features, train_labels, theta, theta_0)
# val_accuracy = lab2.accuracy(val_final_features, val_labels, theta, theta_0)

# print("Custom features")
# print("Training accuracy: {:.4f}".format(train_accuracy))
# print("Validation accuracy: {:.4f}".format(val_accuracy))

#-------------------------------------------------------------------------------
# Part 4 - Testing the Model
#-------------------------------------------------------------------------------

# theta, theta_0 = lab2.perceptron(train_final_features, train_labels, T_best)

# train_accuracy = lab2.accuracy(train_final_features, train_labels, theta, theta_0)
# val_accuracy = lab2.accuracy(val_final_features, val_labels, theta, theta_0)
# test_accuracy = lab2.accuracy(test_final_features, test_labels, theta, theta_0)

# print("Training accuracy: {:.4f}".format(train_accuracy))
# print("Validation accuracy: {:.4f}".format(val_accuracy))
# print("Test accuracy: {:.4f}".format(test_accuracy))
