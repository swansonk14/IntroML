# https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py
# https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

from keras import backend as K

import lab9
import utils

#-------------------------------------------------------------------------------
# Part 1 - Fully Connected Neural Network
#-------------------------------------------------------------------------------

# Set hyperparameters
batch_size = 128
epochs = 10
num_classes = 10
input_shape = (784,)

# Load MNIST data
x_train, y_train, x_test, y_test = utils.load_mnist_fc()
print('{} train samples'.format(x_train.shape[0]))
print('{} test samples'.format(x_test.shape[0]))
print()

# Build model
model = lab9.build_model_fc(input_shape, num_classes)
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Visualize model prior to training
utils.display_activation_maps(model, x_test[0], x_test[0].reshape(28, 28))

# Train model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# Test model
score = model.evaluate(x_test, y_test, verbose=0)
print()
print('Test loss = {}'.format(score[0])) # 0.0957
print('Test accuracy = {}'.format(score[1])) # 0.9818

# Visualize model after training
utils.display_activation_maps(model, x_test[0], x_test[0].reshape(28, 28))

#-------------------------------------------------------------------------------
# Part 2 - Convolutional Neural Network
#-------------------------------------------------------------------------------

# Set hyperparameters
batch_size = 128
epochs = 2
num_classes = 10
img_rows, img_cols = 28, 28

# Set data format based on backend
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)

# Load MNIST data
x_train, y_train, x_test, y_test = utils.load_mnist_conv()
print('{} train samples'.format(x_train.shape[0]))
print('{} test samples'.format(x_test.shape[0]))
print()

# Build model
model = lab9.build_model_conv(input_shape, num_classes)
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

# Visualize model prior to training
utils.display_activation_maps(model, x_test[0], x_test[0][:,:,0])

# Train model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# Test model
score = model.evaluate(x_test, y_test, verbose=0)
print()
print('Test loss = {}'.format(score[0])) # 0.0400
print('Test accuracy = {}'.format(score[1])) # 0.9859

# Visualize model after training
utils.display_activation_maps(model, x_test[0], x_test[0][:,:,0])
