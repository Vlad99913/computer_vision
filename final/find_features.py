from imutils import face_utils
import dlib
import cv2 as cv
import numpy as np


def find_face_features(gray1, gray2):
    """
    Find the facial correspondences between 2 images
    :param gray1: input grayscale image
    :param gray2: input grayscale image
    :return: lists of correspondent feature points
    """

    # Load facial features detector
    path = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(path)

    # detect the faces present in each image
    rects1 = detector(gray1)
    rects2 = detector(gray2)

    # Predict the facial features and transform them to numpy arrays
    shape1 = []
    for i, rect in enumerate(rects1):
        shape1 = predictor(gray1, rect)
        shape1 = face_utils.shape_to_np(shape1)

    shape2 = []
    for i, rect in enumerate(rects2):
        shape2 = predictor(gray2, rect)
        shape2 = face_utils.shape_to_np(shape2)
    return shape1, shape2


def find_features(gray1, gray2):
    """
    Find keypoint correspondences between 2 images
    :param gray1: input grayscale image
    :param gray2: input grayscale image
    :return: matching keypoints
    """
    # find the keypoints and descriptors with SIFT
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    return pts1, pts2


if __name__ == '__main__':
    img1 = cv.imread('images/dt.jpg')
    img2 = cv.imread('images/dt_mirror.jpg')

    gray1 = cv.imread('images/dt.jpg', 0)
    gray2 = cv.imread('images/dt_mirror.jpg', 0)
    #shape1, shape2 = find_face_features(gray1, gray2)
    # Draw on our image, all the found coordinate points (x,y)
    pts1, pts2 = find_features(gray1, gray2)


    for (x, y) in pts1:
        cv.circle(img1, (x, y), 2, (255, 0, 0), -1)
    cv.imwrite("face_features.jpg", img1)
    # write the result
    #cv.imwrite('images/einstein1_features.jpg', img1)

    #for (x, y) in shape2:
     #   cv.circle(img2, (x, y), 2, (0, 255, 0), -1)

    # write the result
    #cv.imwrite('images/einstein2_features.jpg', img2)