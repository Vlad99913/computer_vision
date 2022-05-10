import numpy as np
import cv2 as cv
import glob


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# List of 3D points with Z constant (= 0)
a = 9
b = 6
objp = np.zeros((a*b,3), np.float32)
objp[:,:2] = np.mgrid[0:a,0:b].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
fname = "check7.jpg"

img = cv.imread(fname)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print(gray.shape)

# Find the chess board corners
ret, corners = cv.findChessboardCorners(gray, (a,b), cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_FILTER_QUADS)
# If found, add object points, image points (after refining them)
if ret == True:
    objpoints.append(objp)
    corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners)
    # Draw and save the corners
    cv.drawChessboardCorners(img, (a,b), corners2, ret)
    cv.imwrite("C:/Users/vstef/Desktop/MA2/computer vision/assignment1/res_check7.jpg",img)

    #Obtain intrinsic parameters of camera
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Camera matrix")
    print(mtx)
    print("Distortion")
    print(dist)


