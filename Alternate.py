from time import time

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.layers import Dropout, Flatten
from tensorflow.python.keras.applications.resnet import ResNet50, layers
from tensorflow.python.keras.utils.version_utils import callbacks

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Convolutional Neural Network

# Importing the libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(tf.__version__)

# Part 1 - Data Preprocessing

# Preprocessing the Training set
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=260,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='reflect',
    validation_split=0.2
)
training_set = train_datagen.flow_from_directory('Datasets/Train',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=260,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='reflect',
    validation_split=0.2
)
test_set = test_datagen.flow_from_directory('Datasets/Test',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')
print(type(test_set))
print(type(test_datagen))
from tensorflow.keras.layers import Dense

resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
resnet_base.summary()

# Part 2 - Building the CNN
# Initialising the CNN
cnn = tf.keras.models.Sequential()
cnn.add(resnet_base)
cnn.add(Flatten())
cnn.add(Dropout(0.4))
cnn.add(Dense(2048, activation='relu'))
cnn.add(Dense(4, activation='softmax'))
input_shape = (64, 64, 3)
cnn.build(input_shape)
cnn.summary()
learning_rate = 1e-4
cnn.compile(optimizer=tf.optimizers.RMSprop(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

print('Fit the model...')
t0 = time()  # timing counter starts
print('The model has started learning...')
nepochs = 50
batch_size = 32
history = cnn.fit(training_set,  # Learning process starts
                  steps_per_epoch=4000 // batch_size,
                  epochs=nepochs,
                  validation_data=training_set,
                  validation_steps=2217 // batch_size,
                  callbacks=callbacks)
print('Fit model took', int(time() - t0), 's')  # time is calculated with the help of counter
# Image Segmentation Code

import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_axes([0, 0, 2, 2])
path = 'agdhi/black_seeds/{}'.format('img_1629537671374.jpg_ext0.jpg')
plt.imshow(mpimg.imread(path))
from skimage import data

import matplotlib.pyplot as plt

# Setting the plot size to 15,15
plt.figure(figsize=(15, 15))
coffee = mpimg.imread(path)
gray_coffee = rgb2gray(coffee)
# Setting the plot size to 15,15
plt.figure(figsize=(15, 15))
for i in range(1, 4):
    # Iterating different thresholds
    binarized_gray = (gray_coffee > i * 0.1) * 1
    plt.subplot(5, 2, i + 1)
    # Rounding of the threshold
    # value to 1 decimal point
    plt.title("Threshold: >" + str(round(i * 0.1, 1)))
    # Displaying the binarized image
    # of various thresholds
    plt.imshow(binarized_gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
plt.tight_layout()
plt.show()
