# import the necessary packages
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,required=True,
	help="file to be predicted")
ap.add_argument("-m", "--model", type=str, default="covid19.model",
	help="model to be used to predict image")
args = vars(ap.parse_args())

data = []

# load the image, swap color channels, and resize it to be a fixed
# 224x224 pixels while ignoring aspect ratio
image = cv2.imread(args['image'])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))

data.append(image)

# convert the data and labels to NumPy arrays while scaling the pixel
# intensities to the range [0, 255]
data = np.array(data) / 255.0

model = load_model(args['model'])

pred = model(data)

print('Covid-19 : ')
tf.print(pred[0][0])
print('Normal : ')
tf.print(pred[0][1])