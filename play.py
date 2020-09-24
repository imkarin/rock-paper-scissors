import cv2
from keras.models import load_model
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

# rock-paper-scissors classes
categories = {
    0: "rock",
    1: "paper",
    2: "scissors"
}


def load_data():
    def preprocess_image(image, label):
        # Convert [0, 255] range integers to [0, 1] range floats
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image, label

    # Loading the dataset, split into train and test data
    ds_train, info = tfds.load("rock_paper_scissors", with_info=True, split="train", as_supervised=True)
    ds_test = tfds.load("rock_paper_scissors", split="test", as_supervised=True)

    # Repeat dataset forever, shuffle the images, preprocess the images, split by batch
    ds_train = ds_train.repeat().shuffle(1024).map(preprocess_image).batch(32)
    ds_test = ds_test.repeat().shuffle(1024).map(preprocess_image).batch(32)
    return ds_train, ds_test, info


ds_train, ds_test, info = load_data()


def mapper(prediction):
    return categories[prediction]


# load the model
model = load_model("RPSmodel.h5")

# Capture the video
vid = cv2.VideoCapture(0)

while True:
    # Show video capture frame by frame
    ret, frame = vid.read()

    # Rectangle for user to put their hand in
    cv2.rectangle(frame, (100, 100), (400, 400), (255, 255, 255), 2)

    # Extract the image from the rectangle
    roi = frame[100:400, 100:400]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300, 300))
    
    # Classify the hand gesture
    data_sample = next(iter(ds_test))
    sample_image = data_sample[0].numpy()[0]
    prediction = np.argmax(model.predict(img.reshape(-1, *sample_image.shape))[0])
    user_move_name = mapper(prediction)
    print(user_move_name)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video cap object and destroy all windows when the loop ends
vid.release()
cv2.destroyAllWindows()
