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
    for iter in range(10):
        for i in range(4):
            num = random.randint(0, len(src_pts)-1)
            src_pts_list[i] = src_pts[num]
            dst_pts_list[i] = dst_pts[num]

        
        M, mask = cv.findHomography(src_pts_list, dst_pts_list, 0,5.0)
        

        dst = cv.perspectiveTransform(src_pts, M)

        distancias = np.sqrt(np.sum((dst - dst_pts)**2, axis=2))

        distancias = np.sum(distancias)
        
        if distancias < best_distance:
            distancias = best_distance
            best_M = M
            best_Mask = mask

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
        M, mask = ransac(good, kp1, kp2)
        matchesMask = mask.ravel().tolist()
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = matchesMask, # draw only inliers
                        flags = 2)
    img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    plt.imshow(img3, 'gray'),plt.show()
 

if __name__ == "__main__":
    # Llamar a la función main()
    main()


