import cv2 as cv
import numpy as np
import math


def drawlines(img1, img2, lines, pts1, pts2):
    """
    Draw computed epilines
    :param img1: image on which we draw the epilines for the points in img2
    :param img2: image containing the points from which epilines are drawn in img1
    :param lines: corresponding epilines
    :param pts1: points in img1
    :param pts2: points in img2
    """
    r, c = img1.shape
    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


def dist_pt_line(pt, line):
    """
    Compute the distance between a point and a line in 2D space
    :param pt: point coordinates
    :param line: (a, b, c) characterizing line ax + by + c = 0
    :return: distance bewteen the  line and the point
    """
    x, y = pt
    a, b, c = line
    nom = abs(a*x + b*y + c)
    denom = math.sqrt(a**2 + b**2)
    return nom/denom


# Read the images
img1 = cv.imread('left.jpg', 0)  # queryimage # left image
img2 = cv.imread('right.jpg', 0)  # trainimage # right image

# find the keypoints and descriptors with SIFT
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

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

# Compute the fundamental matrix
F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)

# We select only inlier points
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)
cv.imwrite("lines_left.jpg", img5)
cv.imwrite("lines_right.jpg", img3)

# Compute the distance between corresponding points and epipolar lines in img1
distances1 = []
for pt, line in zip(pts1, lines1):
    dist = dist_pt_line(pt, line)
    distances1.append(dist)

# Compute the distance between corresponding points and epipolar lines in img2
distances2 = []
for pt, line in zip(pts2, lines2):
    dist = dist_pt_line(pt, line)
    distances2.append(dist)

# Display the average distance in each image
print("Average distance between epiline and points in left image: ", sum(distances1)/len(distances1))
print("Average distance between epiline and points in right image: ", sum(distances2)/len(distances2))
