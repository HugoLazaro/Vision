import cv2
import numpy as np
from matplotlib import pyplot as plt

def apply(image):
    """
    Apply contrast to an image in gray scale

    :param image: image to apply contrast
    :return: image with contrast
    """ 
    # Transform image to gray
    imgGrey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to enhance the contrast
    imgGrey_eq = cv2.equalizeHist(imgGrey)

    # hist2 = cv2.calcHist([imgGrey],[0], None, [256], [0,256])
    # cumulative_hist2 = np.cumsum(hist2)
    # plt.plot(cumulative_hist2, color='gray')
    # plt.xlabel('intensidad de iluminacion')
    # plt.ylabel('cantidad de pixeles')

    # hist2_eq = cv2.calcHist([imgGrey_eq],[0], None, [256], [0,256])
    # cumulative_hist2_eq = np.cumsum(hist2_eq)
    # plt.figure()
    # plt.plot(cumulative_hist2_eq, color='gray')
    # plt.xlabel('intensidad de iluminacion')
    # plt.ylabel('cantidad de pixeles')
    # plt.show()

    return imgGrey_eq