import cv2
import numpy as np


def apply(image, numColors):
    """
    Reduce the number of colors in an image

    :param image: image to change the number of colors
    :param numColors: number of colors to for the new image to have
    :return: image with the new number of colors
    """ 
    # Color reduction
 
    # reshape the image into a feature vector so that k-means
    # can be applied
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    numClusters = numColors
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,0.85)
    _, labels, centers = cv2.kmeans(pixel_values, numClusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert the centers to uint8 type
    centers = np.uint8(centers)

    # Replace each pixel value with its nearby center value
    resized_img = centers[labels.flatten()]

    # Reshape the image back to its original shape
    resized_img = resized_img.reshape(image.shape)

    return resized_img