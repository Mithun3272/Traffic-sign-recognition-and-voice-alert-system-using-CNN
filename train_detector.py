"""
This is the file that will be used to train a model to detect traffic signs. It will take images from the "Train" folder
and the "images" folder which will contain the random images. Like images/1.jpg, images/2.jpg etc.
"""

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2
from sklearn.model_selection import train_test_split 
from keras.utils import to_categorical 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

# Final h5 file
H5_FILE = "traffic_detector.h5"

# Define the lists for images and labels
data   = []
labels = []

# Get the current working directory
cur_path = os.getcwd() 

# Define valid image extensions
img_ext = ["png", "jpg", "jpeg"]

def load_image(img_path):
    if a.split(".")[-1] not in img_ext:
        # if the extension is not a valid image then return
        return None

    # Load and Resize the Image
    try:
        image = cv2.imread(img_path) 
        image = cv2.resize(image, (30, 30))   
        return image

    except Exception: 
        print("Error loading image") 
    

# First we will load the images of traffic signs with label 1
for i in range(43): 
    path = os.path.join(cur_path, 'train', str(i)) 
    images = os.listdir(path) 
    for a in images:
        image = load_image(os.path.join(path, a))
        # Only Append image has a value
        if image is not None and image.any():
            data.append(image) 
            labels.append(1) 


# Then we will load the random images with label 0
path = os.path.join(cur_path, 'images') 
images = os.listdir(path) 
for a in images:
    image = load_image(os.path.join(path, a))
    # Only Append image has a value
    if image is not None and image.any():
        data.append(image) 
        labels.append(0) 


# Convert the data and labels to numpy arrays
data   = np.array(data)
labels = np.array(labels)

# Splitting training and testing dataset
X_t1, X_t2, y_t1, y_t2 = train_test_split(data, labels, test_size=0.2, random_state=42)

# Print the training and the test split sizes
print("Train Size:", X_t1.shape[0])
print("Test Size:",  X_t2.shape[0])

# Converting the labels into one hot encoding
y_t1 = to_categorical(y_t1, 2)
y_t2 = to_categorical(y_t2, 2)

# Building the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_t1.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(2, activation='softmax'))

# Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model
eps = 5
history = model.fit(X_t1, y_t1, batch_size=32, epochs=eps, validation_data=(X_t2, y_t2))

# Plotting graphs for accuracy
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# Testing accuracy on test split
val_loss, val_accuracy = model.evaluate(X_t2, y_t2)

# Print the validation results
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_accuracy)

# Saving the model as an h5 file
model.save(H5_FILE)