# Rock, paper, scissors
This project contains a CNN model for detecting hand gestures using a rock, paper, scissors dataset. It recognizes and classifies handgestures from the user's camera.


[screenshot]


## Installing and running the application
This documentation assumes that you have Python installed on your computer, as well as a code editor and a webcam available.

1. Clone or download the repository, and then:
```
cd AI-rockpaperscissors
pip install -r requirements.txt
python play.py
```

2. When the program starts, a webcam feed will pop up. Hold your hand in the indicated white square and the application will print the classification of your handgesture ('rock', 'paper' or 'scissors').


## Developer guide
Throughout the entire code, you'll find comments explaining its use and how you can change it. Here, we'll go through the files step by step and explain the code even further.

### train.py
The train.py file start with importing the nessecary packages. After that, we define some hyperparameters:
```
batch_size = 32
num_classes = 3
epochs = 3
```
`num_classes` is the number of categories for the hand gestures, in our case the dataset has 3 categories. `batch_size` is the hyperparameter that defines the number of samples to work through before updating the internal model parameters. `epochs` is the number of times that the learning algorithm will work through the entire training dataset.

Next, we'll define the data loading function:
```
# Loading and preparing the data
def load_data():
    def preprocess_image(image, label):
        # convert [0, 255] range integers to [0, 1] range floats
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image, label

    # loading the CIFAR-10 dataset, splitted between train and test sets
    ds_train, info = tfds.load("rock_paper_scissors", with_info=True, split="train", as_supervised=True)
    ds_test = tfds.load("rock_paper_scissors", split="test", as_supervised=True)

    # repeat dataset forever, shuffle the images, preprocess the images, split by batch
    ds_train = ds_train.repeat().shuffle(1024).map(preprocess_image).batch(batch_size)
    ds_test = ds_test.repeat().shuffle(1024).map(preprocess_image).batch(batch_size)
    return ds_train, ds_test, info
```
In this function, we first define another function `preprocess_image`. It converts the [0, 255] range RGB values of the images to [0, 1] range floats. This is common practice when working with classification models.
After that, we load the TensorFlow dataset in two splits: the train and test set. We preprocess these datasets, loop through their contents (images) with `.map(preprocess_image)` and batch it into chunks, using our earlier defined `batch_size` hyperparameter.

Next, we define what the model looks like:
```
# Create the model

def create_model():
    model = Sequential()
    model.add(AveragePooling2D(6, 3, input_shape=(300, 300, 3)))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # print the summary of the model architecture
    model.summary()
    # training the model using adam optimizer
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

```
The `Sequential` model, according to Keras, consists of a plain stack of layers where each layer has exactly one input tensor and one output tensor. The layers we added are:
1. `AveragePooling2D`: 
2. Two `Conv2D` layers:
3. `MaxPooling2D`:
4. `Dropout(0.5)`:
5. `Flatten()`:
6. Two `Dense` layers:

At the end, we tell the model to show us its summary, and compile it using the 'Adam' optimizer: a commonly used gradient descent method.

Now, we get to load our data and fit the model to it:
```
# load data
ds_train, ds_test, info = load_data()

# execute the create model function
model = create_model()

# callbacks
logdir = os.path.join("logs", "cifar10-model-v1")
tensorboard = TensorBoard(log_dir=logdir)

# create 'results' folder if it doesnt exist yet
if not os.path.isdir("results"):
    os.mkdir("results")

# training the model
model.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=1,
          steps_per_epoch=info.splits["train"].num_examples // batch_size,
          validation_steps=info.splits["test"].num_examples // batch_size,
          callbacks=[tensorboard])
```
We define our training dataset, testing dataset and info using the earlier defined `load_data()` function. We instantiate the model and fit it using this data.

Finally, we load some samples from the 'test' split of the datset, and evaluate how well the model classifies unseen images:
```
# load the test data
# batch_size=-1 to get the full dataset in NumPy arrays from the returned tf.Tensor object
rock_paper_scissors_test = tfds.load(name="rock_paper_scissors", split='test', batch_size=-1)

# tfds.as_numpy makes NumPy array records out of a tf.data.Dataset
rock_paper_scissors_test = tfds.as_numpy(rock_paper_scissors_test)

# separate the x (input image) and y (output label)
x_test, y_test = rock_paper_scissors_test["image"], rock_paper_scissors_test["label"]

# evaluate and save the model
model.evaluate(x_test, y_test)
model.save("RPStest.h5")
```
The evaluation showed us around 80% accuracy, great! We saved the model and it's ready to use in files.


### test.py
The test.py file is a short file created to test the model by letting it predict some samples from the test dataset.

The file starts with importing the same hyperparameters and loading the data, just like the train.py file. After that, we define the categories:
```
# rock-paper-scissors classes
categories = {
    0: "rock",
    1: "paper",
    2: "scissors"
}
```
The order of these are determine by the dataset that we loaded, so don't change it!

Next, we load the model that we saved earlier and pick a sample image from the dataset to predict:
```
# load the model
loaded_model = load_model("RPStest.h5")

# predict an image
data_sample = next(iter(ds_test))
sample_image = data_sample[0].numpy()[0]
sample_label = categories[data_sample[1].numpy()[0]]
prediction = np.argmax(loaded_model.predict(sample_image.reshape(-1, *sample_image.shape))[0])
print("Predicted label:", categories[prediction])
print("True label:", sample_label)
```

Lastly, we use matplotlib to show us the image we've predicted:
```
# show the image
plt.axis('off')
plt.imshow(sample_image)
plt.show()
```

That's it! This was just a quick test, written to see if we could load our model and how well it did on some images. It's not nessecary for the final use of the product. :)

### play.py
Play.py is the file we use to predict live video camera-input with our model. After defining the categories and loading the data like in the previous files, we start our code with:
```
def mapper(prediction):
    return categories[prediction]
```
`mapper` is a simple function that returns the category name that matches the prediction, which is a number ranging from 0 to 2.

Next, we load the model and capture our video-input:
```
# load the model
model = load_model("RPStest.h5")

# Capture the video
vid = cv2.VideoCapture(0)
```
The video is captured using OpenCV's `VideoCapture`. A video is basically a bunch of images (frames) in a row, so we make a loop to run through each of them, as long as the video is on:
```
while True:
    # Show video capture frame by frame
    ret, frame = vid.read()

    # rectangle for user to play
    cv2.rectangle(frame, (100, 100), (400, 400), (255, 255, 255), 2)

    # extract the image within the user's rectangle
    roi = frame[100:400, 100:400]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300, 300))
    
    # predict the user's gesture
    data_sample = next(iter(ds_test))
    sample_image = data_sample[0].numpy()[0]
    prediction = np.argmax(model.predict(img.reshape(-1, *sample_image.shape))[0])
    user_move_name = mapper(prediction)
    print(user_move_name)


    # Display the resulting frame
    cv2.imshow('frame', frame)

    # press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```
The `cv2.rectangle` draws a rectangle on the screen, for the user to play their hand in. Important to notice is that this rectangle reaches from [100, 100] to [400, 400]: its size is 300 x 300, the same size as the images in our dataset.
We extract the image from the square frame, and convert it to RGB with `cv2.cvtColor`. This is the color-space that our model expects. Just to be sure, we `cv2.resize` our image to 300 x 300. This newly transformed current frame of our video will be called `img`.
To predict our user's gesture, use NumPy to reshape the way our image data is represented. Then we put the newly formatted image data in `model.predict`, which returns a number from 0 to 2. We map a gesture-category name to that number and print this category.
Lastly, we use `cv2.imshow` to give us a pop-up with the output of the current frame. We're basically seeing a live video from our webcams. To stop the loop, press the 'q' button.

That's all the files for now! I hope this documentation was helpful and I will surely keep working on this project going forward. Thank you for reading!