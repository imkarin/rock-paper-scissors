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

# Convert data to numpy arrays, make images grayscale
train_images = np.array([example['image'].numpy()[:, :, 0] for example in ds_train])
train_labels = np.array([example['label'].numpy() for example in ds_train])

test_images = np.array([example['image'].numpy()[:, :, 0] for example in ds_test])
test_labels = np.array([example['label'].numpy() for example in ds_test])

print(test_images.shape)
