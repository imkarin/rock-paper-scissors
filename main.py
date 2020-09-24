# Based on this tutorial: https://www.youtube.com/watch?v=44U8jJxaNp8

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras

tfds.disable_progress_bar()

# Get info from RPS dataset
builder = tfds.builder('rock_paper_scissors')
info = builder.info

# Load in and prepare RPS data
ds_train = tfds.load(name="rock_paper_scissors", split="train")
ds_test = tfds.load(name="rock_paper_scissors", split="test")

# Data preparation: convert data to numpy arrays, make images grayscale ---------------------------------------
train_images = np.array([example['image'].numpy()[:, :, 0] for example in ds_train])
train_labels = np.array([example['label'].numpy() for example in ds_train])

test_images = np.array([example['image'].numpy()[:, :, 0] for example in ds_test])
test_labels = np.array([example['label'].numpy() for example in ds_test])
# print(test_images.shape) now has a length of 3, we need 4...

# Reshape the image data: tell the network that it's a grayscale image (color channel to 1)
train_images = train_images.reshape(2520, 300, 300, 1)
test_images = test_images.reshape(372, 300, 300, 1)

# print(train_images.dtype) currently returns uint8, we want float32:
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

# Max RGB-value for a pixel is 255, so scale every value to be between 0 and 1 (easier for the network to learn)
train_images /= 255
test_images /= 255

# Train the network (the basic way: fully connected neural network) --------------------------------------
# Define the layers of the model: size of layer, activation...
model = keras.Sequential([
    keras.layers.Flatten(),  # transform 300 by 300 images into a single column
    keras.layers.Dense(512, input_shape=(300, 300, 1), activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(3, activation='softmax')  # output layer. softmax is good for classification identifying 1 label
])

# Loss function
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Fit our data to the model
model.fit(train_images, train_labels, epochs=5, batch_size=32)
