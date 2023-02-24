import cv2 # Import python-supported OpenCV functions
import numpy as np # Import numpy and call it np
from matplotlib import pyplot as plt # Import pyplot and call it plt

"""
# initialize the camera
# If you have multiple camera connected with 
# current device, assign a value in cam_port 
# variable according to that
cam_port = -1
cam = cv2.VideoCapture(cam_port)

# reading the input using the camera
result, image = cam.read()

# If image will detected without any error, 
# show result
if result:

    # showing result, it take frame name and image 
    # output
    cv2.imshow("GeeksForGeeks", image)

    # saving image in local storage
    cv2.imwrite("GeeksForGeeks.png", image)

    # If keyboard interrupt occurs, destroy image 
    # window
    cv2.waitKey(0)
    cv2.destroyWindow("GeeksForGeeks")

# If captured image is corrupted, moving to else part
else:
    print("No image detected. Please! try again")

del(cam)
"""


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

"""
# Color reduction
# Load de image
img = cv2.imread('lena.jpg')

# reshape the image into a feature vector so that k-means
# can be applied
pixel_values = img.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# apply k-means using the specified number of clusters and
# then create the quantized image based on the predictions
numClusters = 4
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
"""

"""
# Contrast and equalization
imgGrey = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('lena.jpg')
#cv2.namedWindow( 'Example1', cv2.WINDOW_AUTOSIZE )

cv2.imshow('OriginalGrey',imgGrey)

# Apply histogram equalization to enhance the contrast
imgGrey_eq = cv2.equalizeHist(imgGrey)
cv2.imshow('Enhanced', imgGrey_eq)

#hist(images, channels, mask, histSize, ranges hist)
# color = ('b', 'g', 'r')

# for i, c in enumerate(color):
#     hist = cv2.calcHist([img],[i], None, [256], [0,256])
#     plt.plot(hist, color=c)

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
#cv2.destroyWindow( 'Example1' ) 
cv2.destroyAllWindows()
"""