# IntroML Labs

These Python coding labs are intended to give you a flavor of the incredibly cool things machine learning can do in practice.

## Table of Contents

* [Completing the Labs](#completing-the-labs)
* [Lab 1 - Setup and Data Loading](#lab-1---setup-and-data-loading)
* [Lab 2 - Perceptron Algorithm](#lab-2---perceptron-algorithm)
* [Lab 3 - Support Vector Machines](#lab-3---support-vector-machines)
* [Lab 4 - Non-Linear Classifiers and Kernels](#lab-4---non-linear-classifiers-and-kernels)
* [Lab 5 - Random Forest Classifiers](#lab-5---random-forest-classifiers)
* [Lab 6 - Recommender Systems](#lab-6---recommender-systems)
* [Lab 7 - Neural Networks I](#lab-7---neural-networks-i)
* [Lab 8 - Neural Networks II](#lab-8---neural-networks-ii)
* [Lab 8.5 - Universal Approximation Theorem](#lab-85---universal-approximation-theorem)
* [Lab 9 - Convolutional Neural Networks](#lab-9---convolutional-neural-networks)
* [Lab 10 - Unsupervised Learning](#lab-10---unsupervised-learning)
* [Lab 11 - Reinforcement Learning](#lab-11---reinforcement-learning)

## Completing the Labs

In every lab, you will find `labX.pdf`, which gives instructions about how to to complete that lab. In general, your task is to write the code to complete the functions defined in `labX.py`. After completing each portion of the lab, you should uncomment the relevant section in `main.py` and then run `python main.py`. This will import the functions you wrote in `labX.py` and test their functionality.

## Lab 1 - Setup and Data Loading

This lab provides instructions for installing Python and the relevant Python packages needed in the rest of the labs. It also gives some practice loading and plotting data points.

## Lab 2 - Perceptron Algorithm

In this lab, you will implement the Perceptron Algorithm to build a linear classifier for classifying Amazon product reviews as positive or negative.

## Lab 3 - Support Vector Machines

This lab explores Support Vector Machines (SVMs), which are a type of maximum-margin linear classifier. You will first implement the Pegasos Algorithm, which is an online version of SVM, and you will then compare it's performance to scikit-learn's offline implementation of SVM. You will also learn how to do a grid search to find the best parameters for a machine learning model.

## Lab 4 - Non-Linear Classifiers and Kernels

Since there are many data sets with distributions that cannot be accurately classified by a linear classifier, this lab explores non-linear classifiers and kernels, which make accurate classifications possible on such data sets.

## Lab 5 - Random Forest Classifiers

In this lab, you will learn how to build and train a Random Forest to distinguish between benign and malignant breast cancer tumors based on a number of features such as size and texture. You will also visualize the Random Forest that is learned, illustrating one method of interpreting a machine learning model.

## Lab 6 - Recommender Systems

This lab explores several machine learning models for recommending movies based on user reviews.

## Lab 7 - Neural Networks I

This is the first part of a two-part lab in which you will build your own neural network. In this part, you will build the network's parameters and implement the forward prediction pass of the network.

## Lab 8 - Neural Networks II

This is the second part of the neural networks lab that you began in Lab 7. In this part, you will implement backpropagation and train your network. You will also implement a neural network using Keras.

## Lab 8.5 - Universal Approximation Theorem

Although the Universal Approximation Theorem says that a single hidden-layer neural network can approximate almost any function, this lab explores how well these networks can approximate in practice. Futhermore, this lab explores the ability of single layer networks and deep networks to generalize beyond the range of data they are trained on.

## Lab 9 - Convolutional Neural Networks

In this lab, you will use Keras to implement a Convolutional Neural Network to classify images of hand-written digits.

## Lab 10 - Unsupervised Learning

This lab explores several applications of unsupervised learning including:

* Dimensionality reduction using PCA and t-SNE to visualize high-dimensional data in two dimensions
* Clustering using the k-means clustering algorithm to compress images
* Image compression and denoising using an autoencoder
* Image generation using a variational autoencoder

## Lab 11 - Reinforcement Learning

In this lab, you will use reinforcement learning to learn how to play a simple game.
