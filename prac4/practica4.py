import os
import cv2
import numpy as np

# SIFT, HARRIS, ORB y AKAZE

"""
https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
"""

IMAGES_PATH = "./BuildingScene/"

ALGO = ["ORB", "SIFT", "AKAZE"]

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
nfeatures = 0
image_paths = images_list()
image_1 = cv2.imread(image_paths[1], 0)
image_2 = cv2.imread(image_paths[2], 0)
for algo in ALGO:
    match algo:
        case "HARRIS":
            print("Algoritmo HARRIS")
            # Set Harris detector parameters
            blockSize = 2
            apertureSize = 3
            k = 0.04
            threshold = 0.01
            # Apply Harris corner detector to both images
            corners1 = cv2.cornerHarris(image_1, blockSize, apertureSize, k)
            corners2 = cv2.cornerHarris(image_2, blockSize, apertureSize, k)
            
            # Normalize corner response values
            cv2.normalize(corners1, corners1, 0, 255, cv2.NORM_MINMAX)
            cv2.normalize(corners2, corners2, 0, 255, cv2.NORM_MINMAX)

            # Convert corner response values to uint8
            corners1 = np.uint8(corners1)
            corners2 = np.uint8(corners2)
            
            # Threshold corner response values
            corners1[corners1 < threshold * corners1.max()] = 0
            corners2[corners2 < threshold * corners2.max()] = 0

            # Draw detected corners on original images
            image_1[corners1 > 0] = [0, 0, 255]
            image_2[corners2 > 0] = [0, 0, 255]

            # Display images with detected corners
            cv2.imshow('Image 1 with corners', image_1)
            cv2.imshow('Image 2 with corners', image_2)
            continue
        case "ORB":
            print("Algoritmo ORB")
            detector = cv2.ORB_create(500, 1.2,8,31,0,2)
        case "SIFT":
            print("Algoritmo SIFT")
            detector = cv2.SIFT_create(nfeatures)
        case "AKAZE":
            print("Algoritmo AKAZE")
            detector = cv2.AKAZE_create(nfeatures)
        case _:
            print("Algoritmo de detector de caracteristicas no reconocido")


    # Detectar los puntos de interés y descriptores en la imagen
    keypoints_1, descriptors_1 = detector.detectAndCompute(image_1, None)
    keypoints_2, descriptors_2 = detector.detectAndCompute(image_2, None)

    # Emparejamientos por fuerza bruta
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    # matches es una lista de matches, cada uno es una pareja de descriptores con sus distancias
    matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

    # Apply ratio test
    good_matches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append([m])

    image_matches = cv2.drawMatchesKnn(image_1, keypoints_1, image_2, keypoints_2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    show_image(image_matches, "FB")