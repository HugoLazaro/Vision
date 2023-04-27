import os
import cv2
import numpy as np
import time

# SIFT, HARRIS, ORB y AKAZE

"""
https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
"""

IMAGES_PATH = "./BuildingScene/"

ALGO = ["HARRIS"]

def images_list():
    file_names = os.listdir(IMAGES_PATH)

    # Lista de paths completos a cada imagen
    image_paths = []

    for file_name in file_names:
        image_path = os.path.join(IMAGES_PATH, file_name)
        image_paths.append(image_path)

    return image_paths

def show_image(img, title):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Posibles variables al invocar el script
nfeatures = 1500
image_paths = images_list()
print(image_paths)
image_1 = cv2.imread(image_paths[1], 0)
image1 = cv2.imread(image_paths[1])
image_2 = cv2.imread(image_paths[7], 0)
image2 = cv2.imread(image_paths[7])

#poster, 6
# image_1 = cv2.imread(image_paths[2], 0)
# image_2 = cv2.imread(image_paths[7], 0)

#obj
# image_1 = cv2.imread(image_paths[1], 0)
# image_2 = cv2.imread(image_paths[8], 0)
#obj il
# image_1 = cv2.imread(image_paths[3], 0)
# image_2 = cv2.imread(image_paths[11], 0)

# image_1 = cv2.imread(image_paths[4], 0)
# image_2 = cv2.imread(image_paths[5], 0)

for algo in ALGO:
    match algo:
        case "HARRIS":
            print("Algoritmo HARRIS")
            # Set Harris detector parameters
            threshold = 0.01
        
            # blockSize - It is the size of neighbourhood considered for corner detection
            # ksize - Aperture parameter of the Sobel derivative used.
            # k - Harris detector free parameter in the equation.
            blockSize = 2
            k = 3
            ksize = 0.02
            
            tIni = time.time()
            dst = cv2.cornerHarris(image_1,blockSize,k,ksize)
            # result is dilated for marking the corners
            dst = cv2.dilate(dst,None)
            # Threshold for an optimal value
            image1[dst>threshold*dst.max()]=[0,0,255]
            tEnd = time.time()
            print("Tiempo en detectar esquinas en la imagen: " + str(tEnd-tIni))
            print("Numero de pixeles detectados como esquinas en imagen 1: " + str(np.count_nonzero(dst>threshold*dst.max())))


            show_image(image1, 'Image 1 with corners')



            tIni = time.time()
            dst = cv2.cornerHarris(image_2,2,3,0.04)
            #result is dilated for marking the corners, not important
            dst = cv2.dilate(dst,None)
            # Threshold for an optimal value, it may vary depending on the image.
            image2[dst>threshold*dst.max()]=[0,0,255]

            tEnd = time.time()
            print("Tiempo en detectar esquinas en la imagen: " + str(tEnd-tIni))
            print("Numero de esquinas en imagen 2: " + str(np.count_nonzero(dst>threshold*dst.max())))

            show_image(image2, 'Image 2 with corners')
            
            continue

        case "ORB":
            print("Algoritmo ORB")
            #detector = cv2.ORB_create(500, 1.2,8,31,0,2)
            # nfeatures: This is the maximum number of features to retain. The default value is 500

            # scaleFactor: This parameter compensates for different levels of image blurring by building an image pyramid of multiple resolutions. The default value is 1.2
            # nlevels: This is the number of levels in the image pyramid. The default value is 8
            
            # edgeThreshold: This is the threshold for rejecting weak features along the edges. The default value is 31
            
            # firstLevel: This is the index of the level in the pyramid where the input image is stored. The default value is 0
            
            # WTA_K: This parameter specifies the number of points to sample for building a binary descriptor. The default value is 2
            # scoreType: This parameter specifies the scoring algorithm used to rank features. The default value is cv2.ORB_HARRIS_SCORE
            # patchSize: This parameter specifies the size of the patch used to build a descriptor. The default value is 31
            # fastThreshold: This parameter specifies the threshold used by the FAST algorithm to detect corners. The default value is 20
                # Increasing this parameter can potentially detect keypoints in regions with higher contrast variations, but it may also result in detecting false keypoints in noisy or low-contrast regions.
            detector = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=2, edgeThreshold=15, fastThreshold=10)
        case "SIFT":
            print("Algoritmo SIFT")
            detector = cv2.SIFT_create()
        case "AKAZE":
            print("Algoritmo AKAZE")
            detector = cv2.AKAZE_create()
            # image_1 = cv2.resize(image_1, (0, 0), fx=2, fy=2)
            # image_2 = cv2.resize(image_2, (0, 0), fx=2, fy=2)
        case _:
            print("Algoritmo de detector de caracteristicas no reconocido")


    # Detectar los puntos de interés y descriptores en la imagen
    tIni = time.time()
    keypoints_1, descriptors_1 = detector.detectAndCompute(image_1, None)
    keypoints_2, descriptors_2 = detector.detectAndCompute(image_2, None)
    tEnd = time.time()
    print("Tiempo en detectar caracteristicas de las imagenes: " + str(tEnd-tIni))
    
    print("Numero de caracteristicas en imagen 1: " + str(len(keypoints_1)))
    print("Numero de caracteristicas en imagen 2: " + str(len(keypoints_2)))

    # Emparejamientos por fuerza bruta
    # BFMatcher with default params
    tIni = time.time()
    
    bf = cv2.BFMatcher()
    # matches es una lista de matches, cada uno es una pareja de descriptores con sus distancias
    matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

    # Apply ratio test
    good_matches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append([m])

    tEnd = time.time()
    print("Tiempo en emparejar caracteristicas: " + str(tEnd-tIni))
    print("Numero de emparejamientos: " + str(len(good_matches)))

    image_matches = cv2.drawMatchesKnn(image_1, keypoints_1, image_2, keypoints_2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    show_image(image_matches, "FB")