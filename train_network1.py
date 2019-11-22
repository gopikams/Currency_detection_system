#!/usr/bin/env python3

# import the libraries
import matplotlib
matplotlib.use("Agg")

# import more libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from py.lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())
# intialize the number of epochs to train for, intial learnng rate,and batch size
EPOCHS = 25
INIT_LR = 1e-3
BS = 32
#intialize the data and labels

print("[INFO] loading images...")
data = []
labels = []
# grab the image ad randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)
# lopp over the input images
for imagePath in imagePaths:
  # load the image ,pre-process it , and store it to the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (28, 28))
    image = img_to_array(image)
    data.append(image)
# extract the class label from the image path and update the label list
    label = imagePath.split(os.path.sep)[-2]
    
    if label == "Unknown":
        label = 0
	
    elif label == "Ten":
        label = 1	

    elif label == "Hundred":
        label = 2
   
    elif label == "Thousand":
        label = 3

    labels.append(label)
# scale the raw pixel intensities to the range[0,1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
# partition the data ito training and testing splits (75%-training 25% testing)
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)
#convert the labels from integers to vectors

trainY = to_categorical(trainY, num_classes=4)
testY = to_categorical(testY, num_classes=4)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# ntialize the model
print("[INFO] compiling model...")
model = LeNet.build(width=28, height=28, depth=3, classes=4)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])


