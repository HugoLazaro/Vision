import cv2 # Import python-supported OpenCV functions
import numpy as np # Import numpy and call it np
from matplotlib import pyplot as plt # Import pyplot and call it plt
import os

# Color reduction
# Load de image
img = cv2.imread('GeeksForGeeks.png')

# reshape the image into a feature vector so that k-means
# can be applied
pixel_values = img.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# apply k-means using the specified number of clusters and
# then create the quantized image based on the predictions
numClusters = 2
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,0.85)
_, labels, centers = cv2.kmeans(pixel_values, numClusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert the centers to uint8 type
centers = np.uint8(centers)

# Replace each pixel value with its nearby center value
resized_img = centers[labels.flatten()]

# Reshape the image back to its original shape
resized_img = resized_img.reshape(img.shape)

# display the images and wait for a keypress
cv2.imshow("image", np.hstack([img, resized_img]))
cv2.waitKey(0)
cv2.destroyAllWindows()