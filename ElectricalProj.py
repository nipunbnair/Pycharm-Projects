from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# tensorflow is google's image processing database
# it makes image processing easier
# keras is a high level API to build and train models in tensorflow
# keras is also used for fast prototyping,advanced research and production

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} ({}))".format(class_names[predicted_label],
                                 100 * np.max(predictions_array),
                                 color=color))


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="cyan")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('green')


# version of TensorFlow
print(tf.__version__)
# importing data images and storing in variable
fashion_mnist = keras.datasets.fashion_mnist
# differentiating between images used for training and images used for testing
# out of the 70,000 images in package,60,000 are used for training and 10,000 are used for testing
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# different categories these images will be differentiated into
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# regularization of images
train_images = train_images / 255.0
# train_labels = train_labels / 255.0
# Display all images in training set and display class names below it.
plt.figure(figsize=(10, 10))
for i in range(25):
    print(train_labels[i], train_images.shape, train_labels.shape, type(train_labels))
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Building Model
# Flatten Layer changes 2D picture of 28X28 into a 1D Array of 784 pixels
# 2 Dense layers are present
# the first Dense layer has 128 nodes used for calculation
# the second Dense layer has 10 nodes which returns 10 probability scores
# The probability scores represent probability image belongs to that particular class
# 10 probabilities received sum to 1
# Each node consists of a score current image belongs to one of those 10 classes

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
# Compiling of model
# Loss function measures accuracy of model it predicts error in NN
# optimizer works based on the data obtained and the loss function
# Metrics monitors training and modelling steps
# Accuracy is defined as the fraction of images that are classified
# train the model with train_images and train_label
# test the model using test_images and test_labels

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# epochs means a particular division of time
# number of times the learning algorithm will work through the entire training set

model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test Accuracy', test_acc)
# make predictions on unseen images
predictions = model.predict(test_images)
# predicts the probability of the image being part of class
# returns 10X1 Array which contains the probability  of image being in each class
print(predictions[0])
# the actual answer /correct class
np.argmax(predictions[0])
# the answer given by model
print(test_labels[0])
# predicting image of i=0 and plotting the accuracy in graph
i = 0  # Change value of i
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.show()
# predicting a range of images by passing them through NN
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, test_labels)
plt.show()
# print shape of test image
img = test_images[0]
print(img.shape)
# get dimensions of image
img = (np.expand_dims(img, 0))
print(img.shape)
# getting prediction matrix of image
predictions_single = model.predict(img)
print(predictions_single)
# plot probability graph based on the array
plot_value_array(0, predictions_single, test_labels)
plt.xticks(range(10), class_names, rotation=45)
# print final accuracy of image out of 10
print(np.argmax(predictions_single[0]))
