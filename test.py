import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

# Setting hyper-parameters
batch_size = 32
num_classes = 3
epochs = 3


def load_data():
    def preprocess_image(image, label):
        # convert [0, 255] range integers to [0, 1] range floats
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image, label

    # loading the dataset, split into train and test
    ds_train, info = tfds.load("rock_paper_scissors", with_info=True, split="train", as_supervised=True)
    ds_test = tfds.load("rock_paper_scissors", split="test", as_supervised=True)

    # repeat dataset forever, shuffle the images, preprocess the images, split by batch
    ds_train = ds_train.repeat().shuffle(1024).map(preprocess_image).batch(batch_size)
    ds_test = ds_test.repeat().shuffle(1024).map(preprocess_image).batch(batch_size)
    return ds_train, ds_test, info


ds_train, ds_test, info = load_data()

# rock-paper-scissors classes
categories = {
    0: "rock",
    1: "paper",
    2: "scissors"
}

# load the model
loaded_model = load_model("RPSmodel.h5")

# predict an image
data_sample = next(iter(ds_test))
sample_image = data_sample[0].numpy()[0]
sample_label = categories[data_sample[1].numpy()[0]]
prediction = np.argmax(loaded_model.predict(sample_image.reshape(-1, *sample_image.shape))[0])
print("Predicted label:", categories[prediction])
print("True label:", sample_label)

# show the image
plt.axis('off')
plt.imshow(sample_image)
plt.show()
