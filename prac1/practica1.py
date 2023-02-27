import cv2 # Import python-supported OpenCV functions
import numpy as np # Import numpy and call it np
from matplotlib import pyplot as plt # Import pyplot and call it plt
import os
from efectos import distorsion
from efectos import contraste
from efectos import alien
from efectos import poster

def show_images(img1, img2):
    # Display the original and filtered images side by side
    cv2.imshow("Image", np.hstack([img1, img2]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#img = cv2.imread('lena.jpg')
#Capture an image from the webcam
cam_port = 0
cam = cv2.VideoCapture(cam_port)

# reading the input using the camera
result, img = cam.read()

# If image will detected without any error, 
# show result
if not result:
    print("No image detected. Please! try again")

del(cam)


print("Press 1 to apply contraste filter")
print("Press 2 to apply alien filter")
print("Press 3 to apply poster filter")
print("Press 4 to apply distorsion filter")

# Take user input for filter selection
filter_choice = input("Enter your filter choice: ")

# Apply the selected filter to the image
if filter_choice == "1":
    filtered_img = contraste.apply(img)
    show_images(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), filtered_img)
elif filter_choice == "2":
    filtered_img = alien.apply(img)
    show_images(img, filtered_img)
elif filter_choice == "3":
    numColors = input("Number of colors the new image should have: ")
    filtered_img = poster.apply(img, int(numColors))
    show_images(img, filtered_img)
elif filter_choice == "4":
     #filtered_img = apply_filter(img, "grayscale")
    k1 = input("Distorsion coefficient 1 to apply to the image: ")
    k2 = input("Distorsion coefficient 2 to apply to the image: ")
    filtered_img = distorsion.apply(img, float(k1), float(k2))
    show_images(img, filtered_img)
else:
    print("Invalid choice, no filter applied")
    filtered_img = img
