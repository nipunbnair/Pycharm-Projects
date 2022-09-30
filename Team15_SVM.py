from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.layers import Dropout
from tensorflow.python.keras.applications.resnet import ResNet50
from skimage.color import rgb2gray

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
# training_set = rgb2gray(training_set)
# test_set=rgb2gray(test_set)
from tensorflow.keras.layers import Dense

# Part 2 - Building the CNN
# Initialising the CNN
cnn = tf.keras.models.Sequential()
# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, padding="same", kernel_size=3, activation='relu', strides=2,
                               input_shape=[64, 64, 3]))
cnn.add(Dropout(0.2))
# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, padding='same', kernel_size=3, activation='relu'))
cnn.add(Dropout(0.2))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(Dropout(0.2))
# Step 5 - Output Layer
## For Binary Classification
cnn.add(Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='linear'))

cnn.summary()

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer='adam', loss='hinge', metrics=['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
r = cnn.fit(training_set, validation_data=test_set, epochs=50, steps_per_epoch=len(training_set),
            validation_steps=len(test_set))
# plot the loss
import matplotlib.pyplot as plt

plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

# save it as a h5 file
cnn.save('model_r breastcancer.h5')
from tensorflow.keras.models import load_model

# load model
model = load_model('model_r breastcancer.h5')
model.summary()

# Part 4 - Making a single prediction
import numpy as np
from tensorflow.keras.preprocessing import image

test_image = image.load_img('Datasets/Test/0/10261_idx5_x351_y451_class0.png', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = test_image / 255
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
print(result)
if result[0] < 0:
    print("The image classified is not breast cancer")
else:
    print("The image classified is breast cancer")
# Part 4 - Making a single prediction

import numpy as np
from tensorflow.keras.preprocessing import image

test_image = image.load_img('Datasets/Test/1/10261_idx5_x1851_y1051_class1.png', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = test_image / 255
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
print(result)

if result[0] < 0:
    print("The image classified is not breast cancer")
else:
    print("The image classified is breast cancer")
