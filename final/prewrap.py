import cv2 as cv
from find_features import find_face_features, find_features
from experiment import find_F, find_prewarp
import numpy as np


def prewarp_face(gray1, gray2):
    """
    Compute the homography matrices to rectify 2 grayscale images using detected facial features
    :param gray1: input grayscale image
    :param gray2: input grayscale image
    :return: Homography matrices to rectify the images
    """
    # Find feature points
    pts1, pts2 = find_face_features(gray1, gray2)

    # Compute the fundamental matrix
    F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)

    # We keep only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # Compute the homography matrices
    h1, w1 = gray1.shape
    h2, w2 = gray2.shape
    _, H1, H2 = cv.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), F, imgSize=(w1, h1))

    return H1, H2


def prewarp(gray1, gray2):
    """
    Compute the homography matrices to rectify 2 grayscale images
    :param gray1: input grayscale image
    :param gray2: input grayscale image
    :return: Homography matrices to rectify the images
    """
    # Find feature points
    pts1, pts2 = find_features(gray1, gray2)

    # Compute the fundamental matrix
    F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
    # We keep only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # Compute the homography matrices
    h1, w1 = gray1.shape
    h2, w2 = gray2.shape
    _, H1, H2 = cv.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), F, imgSize=(w1, h1))

    return H1, H2


def prewarp_manual(gray1, gray2):
    """
    Compute the homography matrices without using cv.stereoRectifyUncalibrated
    :param gray1: input grayscale image
    :param gray2: input grayscale image
    :return: Homography matrices to rectify the images
    """
    # Compute the fundamental matrix
    F = find_F(gray1, gray2)

    # Compute the homography matrices
    H1, H2 = find_prewarp(F)
    H1 = H1.real.astype(np.float64)
    H2 = H2.real.astype(np.float64)

    return H1, H2


if __name__ == '__main__':
    img1 = cv.imread('images/left.jpg', 0)
    img2 = cv.imread('images/right.jpg', 0)

    img1_color = cv.imread('images/left.jpg')
    img2_color = cv.imread('images/right.jpg')

    h1, w1 = img1.shape
    h2, w2 = img2.shape
    H1, H2 = prewarp(img1, img2)

    prewarp_1 = cv.warpPerspective(img1, H1, (w1, h1))
    prewarp_2 = cv.warpPerspective(img2, H2, (w1, h1))

    cv.imwrite("images/left_prewrap2.jpg", prewarp_1)
    cv.imwrite("images/right_prewrap2.jpg", prewarp_2)
