import cv2 as cv
import numpy as np

# Read both images
img1 = cv.imread("img1.jpg")
img2 = cv.imread("img2.jpg")

# Initiate an ORB detector
orb = cv.ORB.create()

# Find the keypoints and descriptors within each image
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Draw keypoint location without size or orientation
img3 = cv.drawKeypoints(img1, kp1, None, color=(255, 0, 0), flags=0)
img4 = cv.drawKeypoints(img2, kp2, None, color=(0, 255, 0), flags=0)

# Save resulting images
cv.imwrite("img1_kp.png", img3)
cv.imwrite("img2_kp.png", img4)

# Create BFMatcher object
matcher = cv.BFMatcher()

# Use NNDR to match keypoints with their closest neighbour
matches = matcher.knnMatch(des1, des2, k=2)
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

img6 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None,
                         flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imwrite("matches2.png", img6)

# Find the coordinates of matched keypoints
kp1_coord = [kp1[m[0].queryIdx].pt for m in good]
kp2_coord = [kp2[m[0].trainIdx].pt for m in good]
y, x = kp1_coord[0]
y, x = int(y), int(x)

img7 = cv.circle(img1, (y, x), 10, (0, 0, 255), -1)
cv.imwrite("circle.png", img7)

# Compute the translation matrix between both images
N = len(kp1_coord)
A = np.array([[N, 0],
              [0, N]])
b = np.array([[0],
              [0]])
for i in range(N):
    x, y = kp1_coord[i]
    x, y = round(x), round(y)
    x_, y_ = kp2_coord[i]
    x_, y_ = round(x_), round(y_)
    b[0, 0] = b[0, 0] + (x_ - x)
    b[1, 0] = b[1, 0] + (y_ - y)

p = np.linalg.solve(A, b)
print(p)
t_x = round(p[0, 0])
t_y = round(p[1, 0])

m = np.float32([[1, 0, t_x],
                [0, 1, t_y]])
# Translate the images by adding borders
if t_x >= 0:
    img1 = cv.copyMakeBorder(img1, 0, 0, t_x, 0, cv.BORDER_CONSTANT)
    img2 = cv.copyMakeBorder(img2, 0, 0, 0, t_x, cv.BORDER_CONSTANT)
else:
    img1 = cv.copyMakeBorder(img1, 0, 0, 0, abs(t_x), cv.BORDER_CONSTANT)
    img2 = cv.copyMakeBorder(img2, 0, 0, abs(t_x), 0, cv.BORDER_CONSTANT)

if t_y >= 0:
    img1 = cv.copyMakeBorder(img1, t_y, 0, 0, 0, cv.BORDER_CONSTANT)
    img2 = cv.copyMakeBorder(img2, 0, t_y, 0, 0, cv.BORDER_CONSTANT)
else:
    img1 = cv.copyMakeBorder(img1, 0, abs(t_y), 0, 0, cv.BORDER_CONSTANT)
    img2 = cv.copyMakeBorder(img2, abs(t_y), 0, 0, 0, cv.BORDER_CONSTANT)

# Combine both images by taking the average
dst = cv.addWeighted(img1, 0.5, img2, 0.5, 0.0)
cv.imwrite("dst2.png", dst)
