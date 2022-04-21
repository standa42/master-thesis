## code used from: https://stackoverflow.com/questions/11114349/how-to-visualize-descriptor-matching-using-opencv-module-in-python

import numpy as np
import cv2
import os
from config.Config import Config


def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (0, 0, 255), 2)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (0, 0, 255), 2)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 0, 255), 1)


    # Show the image
    cv2.imwrite(os.path.join(Config.DataPaths.DataFolder, "orb_vizualization.png"), out)
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img1 = cv2.imread("./data/unique_rims_collage_dataset/160/2019_05_13_12_11_08_A_frame185_bb0_Wheel.png") # Original image
img2 = cv2.imread("./data/unique_rims_collage_dataset/160/2019_05_13_17_20_56_A_frame149_bb0_Wheel.png") # Rotated image

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Create ORB detector with 1000 keypoints with a scaling pyramid factor
# of 1.2
orb = cv2.ORB_create(1000, 1.2)

# Detect keypoints of original image
(kp1,des1) = orb.detectAndCompute(img1, None)

# Detect keypoints of rotated image
(kp2,des2) = orb.detectAndCompute(img2, None)

# Create matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Do matching
matches = bf.match(des1,des2)

# Sort the matches based on distance.  Least distance
# is better
matches = sorted(matches, key=lambda val: val.distance)

# Show only the top 10 matches
drawMatches(img1, kp1, img2, kp2, matches[:15])




