import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

import random

MIN_MATCH_COUNT = 10

def ransac(matches, kp1, kp2):
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

    src_pts_list = np.zeros((4, 2))
    dst_pts_list = np.zeros((4, 2))

    best_distance = np.Infinity
    best_M, best_Mask = cv.findHomography(src_pts_list, dst_pts_list, 0,5.0)
    for iter in range(100):
        for i in range(4):
            num = random.randint(0, len(src_pts)-1)
            src_pts_list[i] = src_pts[num]
            dst_pts_list[i] = dst_pts[num]


        M, mask = cv.findHomography(src_pts_list, dst_pts_list, 0,5.0)
        # print("Aqui")
        # print(mask.shape)


        dst = cv.perspectiveTransform(src_pts, M)
        #print(len(dst))
        consensus = np.zeros((len(dst),1), dtype=int)
        threshold_distance = 200
        threshold_number = 50
        max_number = 0

        distancias = np.sqrt(np.sum((dst - dst_pts)**2, axis=2))
        for i in range(len(distancias)):
            if distancias[i] < threshold_distance:
                consensus[i] = 1

        if np.sum(consensus) > threshold_number:
            best_M = M
            best_Mask = consensus
            # print(distancias)
            return best_M, best_Mask
        elif np.sum(consensus) > max_number:
            best_M = M
            best_Mask = consensus
            max_number = len(consensus)

        # if distancias < best_distance:
        #     distancias = best_distance
        #     best_M = M
        #     best_Mask = mask

    return best_M, best_Mask





# if len(good)>MIN_MATCH_COUNT:
#     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
#     M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
#     matchesMask = mask.ravel().tolist()
#     h,w = img1.shape
#     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#     dst = cv.perspectiveTransform(pts,M)
#     img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
# else:
#     print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
#     matchesMask = None

# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                    singlePointColor = None,
#                    matchesMask = matchesMask, # draw only inliers
#                    flags = 2)
# img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
# plt.imshow(img3, 'gray'),plt.show()

def main():
    # Código principal del programa
    img1 = cv.imread('BuildingScene/poster1.jpg', cv.IMREAD_GRAYSCALE) # queryImage
    img2 = cv.imread('BuildingScene/poster3.jpg', cv.IMREAD_GRAYSCALE) # trainImage

    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)



    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        # print(M.shape)
        # print(mask.shape)
        # print(mask)

        # M, mask = ransac(good, kp1, kp2)
        # print(M.shape)
        # print(mask.shape)
        # #print(mask)
        matchesMask = mask.ravel().tolist()
        h, w = img1.shape

        img1_transformed = cv.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))
        
        cv.imshow("transformed",img1_transformed)
        cv.waitKey(0)

        cv.imshow("img1",img1)
        cv.waitKey(0)
        #plt.imshow(img1_transformed, 'gray'),plt.show()

        # img1_resized = cv.resize(img1, (img2.shape[1] + img1.shape[1], img2.shape[0] + img1.shape[0]))

        # img2_resized = cv.resize(img2, (img2.shape[1] + img1.shape[1], img2.shape[0] + img1.shape[0]))
        # # Juntar las dos imágenes
        # result = cv.addWeighted(img2_resized, 0.5, img1_resized, 0.5, 0)

        
        result = cv.addWeighted(img2, 0.5, img1_transformed, 0.5, 0)
        #plt.imshow(result, 'gray'),plt.show()
        cv.imshow("weigh",result)
        cv.waitKey(0)



        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        
        dst = cv.perspectiveTransform(pts,M)
        
        h, w = img2.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        print("pts")
        print(pts)
        dst = np.concatenate((pts,dst),axis=0)
        #dst = dst.concat(pts)
        print("dst")
        print(dst)
        minimos = np.amin(dst, axis=0, out=None)
        maximos = np.amax(dst, axis=0, out=None)
        # print(minimos)
        # print(maximos)
        dimensiones = maximos - minimos
        print("Dimensiones")
        print(dimensiones)  
        
        
        
        #matriz traslacion de minimos
        matrix = np.array([[1.0,0.0,-minimos[0][0]],
                           [0.0,1.0,-minimos[0][1]],
                           [0.0,0.0,1.0]
                           ])
        print(matrix)
        print(M)
        print(np.dot(matrix,M))
        img1_transformed_2 = cv.warpPerspective(img1, np.dot(matrix,M), (int(dimensiones[0][0]), int(dimensiones[0][1])))

        img2_transformed_2 = cv.warpPerspective(img2, matrix, (int(dimensiones[0][0]), int(dimensiones[0][1])))

        
        cv.imshow("img_1_transformed_2",img1_transformed_2)
        cv.waitKey(0)

        cv.imshow("img_2_transformed_2",img2_transformed_2)
        cv.waitKey(0)

        final = np.maximum(img1_transformed_2, img2_transformed_2)
        cv.imshow("final",final)
        cv.waitKey(0)


        img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = matchesMask, # draw only inliers
                        flags = 2)
    img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    cv.imshow("img3",img3)
    cv.waitKey(0)


if __name__ == "__main__":
    # Llamar a la función main()
    main()


