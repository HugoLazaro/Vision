import cv2 # Import python-supported OpenCV functions
import numpy as np # Import numpy and call it np
from matplotlib import pyplot as plt # Import pyplot and call it plt
import os

def apply(image):
    """
    Apply contrast to an image in gray scale

    :param image: image to apply contrast
    :return: image with contrast
    """ 
    # Transform image to gray
    imgGrey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('OriginalGrey',imgGrey)

    # Apply histogram equalization to enhance the contrast
    imgGrey_eq = cv2.equalizeHist(imgGrey)
    #cv2.imshow('Enhanced', imgGrey_eq)


    hist2 = cv2.calcHist([imgGrey],[0], None, [256], [0,256])
    cumulative_hist2 = np.cumsum(hist2)
    plt.plot(cumulative_hist2, color='gray')
    plt.xlabel('intensidad de iluminacion')
    plt.ylabel('cantidad de pixeles')

    hist2_eq = cv2.calcHist([imgGrey_eq],[0], None, [256], [0,256])
    cumulative_hist2_eq = np.cumsum(hist2_eq)
    plt.figure()
    plt.plot(cumulative_hist2_eq, color='gray')
    plt.xlabel('intensidad de iluminacion')
    plt.ylabel('cantidad de pixeles')
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return imgGrey_eq