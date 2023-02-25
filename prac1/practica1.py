import cv2 # Import python-supported OpenCV functions
import numpy as np # Import numpy and call it np
from matplotlib import pyplot as plt # Import pyplot and call it plt
import os
from efectos import distorsion
from efectos import contraste



""""
# Barrel distorsion
# Load de image
img = cv2.imread('lena.jpg')

#Define distorsion coefficient
k1 = 1 #negative for pincushion

# Center of the image
cx,cy = img.shape[1]/2, img.shape[0]/2

# Focla lenght
fx=img.shape[1]
fy=img.shape[0]

# Distorted image
img_dist = cv2.undistort(
    img,
    np.array([[fx,0,cx],[0,fy,cy],[0,0,1]]),
    #coeff ->k1,k2,k3 radial distorsion deviation of lens from perfect sphere
    #coeff ->p1,p2 tangential distorsion account the offset lens-image sensor -> tilt/shift
    np.array([0, k1, 0, 0, 0]), # distorsion
    None)

cv2.imshow("image", np.hstack([img, img_dist]))
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

#Define distorsion coefficient
# k = -0.5

# Image center
# h,w = img.shape[:2]
# cx,cy = w/2, h/2

# Distorsion map
#initUndistortRectifyMap(cameraMatrix, distCoeff, R, newCamera, size)
# size = (w,h)
# map = cv2.initUndistortRectifyMap(
#     np.array([[1,0,cx],[0,1,cy],[0,0,1]]),
#     np.array([[1,k,0],[0,1,0],[0,0,1]]),
#     size,cv2.CV_32FC1)

# fx=fy=1000

# camera_matrix = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
# distorsion_coefficient = np.array([-0.1, 11,-0.1,0,0])
# new_camera, _ = cv2.getOptimalNewCameraMatrix(
#     camera_matrix, distorsion_coefficient, img.shape[:2], 1, img.shape[:2])
# size = (w,h)

# map = cv2.initUndistortRectifyMap(
#     camera_matrix,
#     distorsion_coefficient,
#     None,
#     new_camera,
#     img.shape[:2],
#     cv2.CV_32FC1
# )

# Apply distorsion to img
# img_dist = cv2.remap(img,map[0],map[1],cv2.INTER_LINEAR)

# # Show the img 
# cv2.imshow("image", np.hstack([img, img_dist]))
# cv2.waitKey(0)
# cv2.destroyAllWindows()






img = cv2.imread('lena.jpg')
# Capture an image from the webcam
# cam_port = 0
# cam = cv2.VideoCapture(cam_port)

# # reading the input using the camera
# result, img = cam.read()

# # If image will detected without any error, 
# # show result
# if not result:
#     print("No image detected. Please! try again")

# del(cam)


# Display a menu for selecting a filter
file_counter = 1
for file in os.listdir("efectos"):
    if file.endswith(".py"):
        print("Press {} to apply {} filter".format(str(file_counter), file[:-3]))
        file_counter = file_counter + 1

# Take user input for filter selection
filter_choice = input("Enter your filter choice: ")

# Apply the selected filter to the image
if filter_choice == "1":
    # Barrel distorsion
    #filtered_img = apply_filter(img, "grayscale")
    k1 = 1
    filtered_img = distorsion.apply(img, k1)
elif filter_choice == "2":
    filtered_img = contraste.apply(img)
# elif filter_choice == "3":
#     filtered_img = apply_filter(img, "edges")
else:
    print("Invalid choice, no filter applied")
    filtered_img = img


# Display the original and filtered images side by side
cv2.imshow("Image", np.hstack([img, filtered_img]))
cv2.waitKey(0)
cv2.destroyAllWindows()


