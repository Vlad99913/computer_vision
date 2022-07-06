import numpy as np
import cv2 as cv
from find_features import find_features


def find_F(gray1, gray2):
    """
    Rectify 2 grayscale images
    :param gray1:
    :param gray2:
    :return: rectified images
    """
    # Find feature points
    pts1, pts2 = find_features(gray1, gray2)

    # Compute the fundamental matrix
    F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
    return F


def find_epipoles(F):
    """
    Calculate epipoles from the fundamental matrix
    Args:
        F: 3x3 fundamental matrix
    Returns
        e0: epipole to image 1
        e1: epipole to image 2
    """
    # Find eigenvalues and eigenvectors of Fand F transposed
    value0, vector0 = np.linalg.eig(F)
    value1, vector1 = np.linalg.eig(np.transpose(F))

    # the epipoles are the eigenvector of the smallest eigenvalue
    e0 = vector0[:, np.argmin(value0)]
    e1 = vector1[:, np.argmin(value1)]

    # print('find epipoles function')
    # print(value0)
    # print(vector0)
    # print(value1)
    # print(vector1)

    return e0, e1


def rotation_matrix(u, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    Args:
        axis:  3X1 numpy array
        theta: scalar. rotation angle
    Returns:
        R: rotation matrix
    """
    c = np.cos(theta)
    s = np.sin(theta)
    t = 1 - np.cos(theta)
    x = u[0]
    y = u[1]
    return np.array([[t * x * x + c, t * x * y, s * y],
                     [t * x * y, t * y * y + c, -s * x],
                     [-s * y, s * x, c]])


def find_prewarp(F):
    """
    Find prewarp transforms H1 and H2
    Args:
        F: fundamental matrix
    Returns:
        H0: prewarp transform for left image
        H1: prewarp transform for right image
    """
    # get epipoles of image 1 and image 2
    e0, e1 = find_epipoles(F)

    # axis of rotation
    d0 = np.array([-e0[1], e0[0], 0])

    # find corresponding axis in image 1
    Fd0 = F.dot(d0)
    d1 = np.array([-Fd0[1], Fd0[0], 0])

    # find angle of rotation
    # theta0 = -np.pi/2 - np.arctan((d0[1]*e0[0] - d0[0]*e0[1])/e0[2])
    # theta1 = -np.pi/2 - np.arctan((d1[1]*e1[0] - d1[0]*e1[1])/e1[2])
    theta0 = np.arctan(e0[2] / (d0[1] * e0[0] - d0[0] * e0[1]))
    theta1 = np.arctan(e1[2] / (d1[1] * e1[0] - d1[0] * e1[1]))

    # change angle to degree
    # theta0 = np.angle(theta0, deg = True)
    # theta1 = np.angle(theta1, deg = True)

    # rotation of angle theta about axis d
    R_d0_theta0 = rotation_matrix(d0, theta0)
    R_d1_theta1 = rotation_matrix(d1, theta1)

    # find new epipoles
    new_e0 = R_d0_theta0.dot(e0)
    new_e1 = R_d1_theta1.dot(e1)

    # find new angle of rotation
    phi0 = -np.arctan(new_e0[1] / new_e0[0])
    phi1 = -np.arctan(new_e1[1] / new_e1[0])

    # change angle to degree
    # phi0 = np.angle(phi0, deg = True)
    # phi1 = np.angle(phi1, deg = True)

    # rotation of angle phi about the zero point
    R_phi0 = np.array([[np.cos(phi0), -np.sin(phi0), 0],
                       [np.sin(phi0), np.cos(phi0), 0],
                       [0, 0, 1]])
    R_phi1 = np.array([[np.cos(phi1), -np.sin(phi1), 0],
                       [np.sin(phi1), np.cos(phi1), 0],
                       [0, 0, 1]])

    H0 = R_phi0.dot(R_d0_theta0)
    H1 = R_phi1.dot(R_d1_theta1)

    return H0, H1


if __name__ == '__main__':
    img1 = cv.imread('images/v1.jpg', 0)
    img2 = cv.imread('images/v2.jpg', 0)

    img1_color = cv.imread('images/v1.jpg')
    img2_color = cv.imread('images/v2.jpg')

    F = find_F(img1, img2)

    H1, H2 = find_prewarp(F)
    H1 = H1.real.astype(np.float64)
    H2 = H2.real.astype(np.float64)
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    prewarp_1 = cv.warpPerspective(img1, H1, (w1, h1))
    prewarp_2 = cv.warpPerspective(img2, H2, (w1, h1))

    cv.imwrite("images/v1_prewarp2.jpg", prewarp_1)
    cv.imwrite("images/v2_prewarp2.jpg", prewarp_2)
