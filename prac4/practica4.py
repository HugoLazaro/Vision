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
nfeatures = 100

for algo in ALGO:
    match algo:
        case "HARRIS":
            pass
        case "ORB":
            detector = cv2.ORB_create(nfeatures)
        case "SIFT":
            detector = cv2.SIFT_create(nfeatures)
        case "AKAZE":
            detector = cv2.AKAZE_create(nfeatures)
        case _:
            print("Algoritmo de detector de caracteristicas no reconocido")
            

    image_paths = images_list()
    keypoints_list = []
    descriptors_list = []

    for path in image_paths:
        image = cv2.imread(path, 0)

        # Detectar los puntos de interés y descriptores en la imagen
        keypoints, descriptors = detector.detectAndCompute(image, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)
        # image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)
        # show_image(image_with_keypoints, "Image with keypoints")


    # # Emparejamientos por fuerza bruta
    # # BFMatcher with default params
    # bf = cv2.BFMatcher()
    # # matches es una lista de matches, cada uno es una pareja de descriptores con sus distancias
    # matches = bf.knnMatch(descriptors_list[1], descriptors_list[2], k=2)

    # # Apply ratio test
    # good_matches = []
    # for m,n in matches:
    #     if m.distance < 0.75*n.distance:
    #         good_matches.append([m])

    # image_matches = cv2.drawMatchesKnn(cv2.imread(image_paths[1], 0), keypoints_list[1], cv2.imread(image_paths[2], 0), keypoints_list[2], good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # show_image(image_matches, "FB")


    # FLANN based Matcher
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(descriptors_list[1],descriptors_list[2],k=2)
    # Need to draw only good matches, so create a mask
        
    good_matches = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good_matches.append([m])

    image_matches = cv2.drawMatchesKnn(cv2.imread(image_paths[1], 0), keypoints_list[1], cv2.imread(image_paths[2], 0), keypoints_list[2], good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    show_image(image_matches, "FLANN")
