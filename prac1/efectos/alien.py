import cv2 # Import python-supported OpenCV functions
import numpy as np # Import numpy and call it np
from matplotlib import pyplot as plt # Import pyplot and call it plt
import os

def apply(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_arr = np.array(hsv)

    # # Define the new skin color as a NumPy array
    new_skin_color = np.array([179, 255, 255])  # HSV values of the new skin color

    # # Define a mask that identifies the skin pixels in the image
    lower_skin_color = np.array([0, 68, 100])  # lower threshold for skin color
    upper_skin_color = np.array([11, 130, 230])  # upper threshold for skin color

    skin_mask = cv2.inRange(img_arr, lower_skin_color, upper_skin_color)
    cv2.imshow("Mascara",skin_mask)
    cv2.waitKey(0)

    # # Replace the skin pixels with the new skin color
    img_arr[skin_mask != 0] = new_skin_color

    img_arrRGB = cv2.cvtColor(img_arr, cv2.COLOR_HSV2BGR)
    return img_arrRGB
