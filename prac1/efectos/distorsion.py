import cv2
import numpy as np


def apply(image, coefficient):
    """
    Apply distorsion to an image

    :param image: image to apply distorsion
    :param coefficient: distorsion coefficient
    :return: image with distorsion
    """ 
    # Center of the image
    cx,cy = image.shape[1]/2, image.shape[0]/2

    # Focla lenght
    fx=image.shape[1]
    fy=image.shape[0]

    # Distorted image
    img_dist = cv2.undistort(
        image,
        np.array([[fx,0,cx],[0,fy,cy],[0,0,1]]),
        #coeff ->k1,k2,k3 radial distorsion deviation of lens from perfect sphere
        #coeff ->p1,p2 tangential distorsion account the offset lens-image sensor -> tilt/shift
        np.array([0, coefficient, 0, 0, 0]), # distorsion
        None)

    return img_dist

