import cv2 # Import python-supported OpenCV functions
import numpy as np # Import numpy and call it np
from matplotlib import pyplot as plt # Import pyplot and call it plt
import os

def apply(image):
    pass

# img = cv2.imread('lena.jpg')

# # Load the image
# #img = cv2.imread("image_path.jpg")
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# img_arr = np.array(hsv)

# # # Define the new skin color as a NumPy array
# new_skin_color = np.array([0, 255, 0])  # RGB values of the new skin color

# # # Define a mask that identifies the skin pixels in the image
# lower_skin_color = np.array([90, 60, 60])  # lower threshold for skin color
# upper_skin_color = np.array([255, 255, 255])  # upper threshold for skin color

# skin_mask = cv2.inRange(img_arr, lower_skin_color, upper_skin_color)

# # # Replace the skin pixels with the new skin color
# img_arr[skin_mask != 0] = new_skin_color

# # Convertir la imagen de BGR a HSV
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# # Definir los rangos de color de la piel en HSV
# lower_skin = np.array([0, 20, 70])
# upper_skin = np.array([20, 255, 255])

# # Crear una máscara de la piel
# mask = cv2.inRange(hsv, lower_skin, upper_skin)

# # Aplicar la máscara a la imagen original
# skin = cv2.bitwise_and(img, img, mask=mask)

# new_skin_color = np.array([0, 255, 0])  # RGB values of the new skin color
# # Cambiar el color de la piel
# skin[:,:,0] += 40
# skin[:,:,1] -= 20

# # Mostrar la imagen con el nuevo color de piel
# cv2.imshow("Imagen con nuevo color de piel", skin)
# cv2.imshow("image", np.hstack([img, img_arr]))
"""
cv2.waitKey(0)
cv2.destroyAllWindows()


# Changes the color of the skin in an image to a new color


#path file
path_face="./paleta.jpg"
result_partial="./result_partial.png"
result_final="./result_partial.png"


#blending parameter
alpha = 0.9


# Define lower and uppper limits of what we call "skin color"
skincolor_low=np.array([0, 48, 78])
skincolor_high=np.array([15, 68, 108])



#specify desired bgr color (brown) for the new face.
#this value is approximated
desired_color_brg = (0, 255, 0)




# read face
img_main_face = cv2.imread(path_face)



# face.jpg has by default the BGR format, convert BGR to HSV
hsv=cv2.cvtColor(img_main_face,cv2.COLOR_BGR2HSV)




#create the HSV mask
mask=cv2.inRange(img_main_face,skincolor_low,skincolor_high)



# Change image to brown where we found pink
img_main_face[mask>0]=desired_color_brg
cv2.imwrite(result_partial,img_main_face)




#blending block start

#alpha range for blending is  0-1


# load images for blending
src1 = cv2.imread(result_partial)
src2 = cv2.imread(path_face)

if src1 is None:
    print("Error loading src1")
    exit(-1)
elif src2 is None:
    print("Error loading src2")
    exit(-1)
    
    
# actually  blend_images
result_final = cv2.addWeighted(src1, alpha, src2, 1-alpha, 0.0)
cv2.imwrite('./result_final.png', result_final)

#blending block end
"""