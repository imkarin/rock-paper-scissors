# Based on this tutorial: https://www.youtube.com/watch?v=44U8jJxaNp8

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from kerastuner.tuners import RandomSearch

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


# Train the network (convolutional) & tune hyperparameters with Kerastuner (finds the best model)
def build_model(hp):
    model = keras.Sequential()

    model.add(keras.layers.AveragePooling2D(6, 3, input_shape=(300, 300, 1)))

    # let the tuner decide amount of convolutional layers (1-3)
    for i in range(hp.Int("Conv Layers", min_value=0, max_value=3)):
        # for every conv layer, let the decide the amount of filters
        model.add(keras.layers.Conv2D(hp.Choice(f"layer_{i}_filters", [16, 32, 64]), 3, activation='relu'))

    model.add(keras.layers.MaxPool2D(2, 2))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Flatten())

    # let the tuner decide density of the dense layer
    model.add(keras.layers.Dense(hp.Choice("Dense layer", [64, 128, 256, 512, 1024]), activation='relu'))

    model.add(keras.layers.Dense(3, activation='softmax'))

    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model


tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=32,
)

tuner.search(train_images, train_labels, validation_data=(test_images, test_labels), epochs=10, batch_size=32)

# Show results of the best model that Kerastuner found, and evaluate the test data
best_model = tuner.get_best_models()[0]
best_model.summary()
best_model.evaluate(test_images, test_labels)