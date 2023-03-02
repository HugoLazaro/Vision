import cv2
import numpy as np


def apply(image, k1, k2):
    """
    Apply distorsion to an image

    :param image: image to apply distorsion
    :param coefficient: distorsion coefficient
    :return: image with distorsion
    """ 
    # Center of the image
    xcen,ycen = image.shape[1]/2, image.shape[0]/2

    # Image size
    fx=image.shape[1]
    fy=image.shape[0]

    print("Image size: " + str(fx) + ", " + str(fy))
    

    new_img = np.zeros((fy,fx,3), np.uint8)
    print("New image size: " + str(new_img.shape[1]) + ", " + str(new_img.shape[0]))
    i = 0

    for yd in range(fy):
        for xd in range(fx):
            r2 = pow((xd - xcen),2) + pow((yd - ycen),2)
            xu = xd + (xd - xcen)* k1 * r2 + (xd - xcen)* k2 * pow(r2,2)
            yu = yd + (yd - ycen)* k1 * r2 + (yd - ycen)* k2 * pow(r2,2)
            xu = int(np.round(xu))
            yu = int(np.round(yu))
            
            if float(xu) < fy and yu < fy and xu > 0 and yu > 0:
                try:
                    new_img[xd][yd] = image[xu][yu]
                except:
                    i = i+1
                    # print("i: " + str(i))
                    # print("fx: " + str(fx))
                    # print("fy: " + str(fy))
                    print("xu: " + str(xu))
                    print("yu: " + str(yu))
                #new_img[xd][yd] = image[xu][yu]

    return new_img
