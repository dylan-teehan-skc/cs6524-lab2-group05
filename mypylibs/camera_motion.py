import cv2 as cv
import numpy as np


def match_keypoints(img1, img2):
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matches = cv.BFMatcher().knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance] 

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2


def relative_pose(pts1, pts2, K):
    E, mask = cv.findEssentialMat(pts1, pts2, K, cv.RANSAC)
    _, R, t, _ = cv.recoverPose(E, pts1, pts2, K, mask=mask)
    return R, t
