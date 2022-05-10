import cv2 as cv

# Read both images
img1 = cv.imread("img1.png", cv.IMREAD_GRAYSCALE)
img2 = cv.imread("img2.png", cv.IMREAD_GRAYSCALE)

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

# Match Keypoints with their nearest neighbor
matches = matcher.knnMatch(des1, des2, k=1)
img5 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None,
                         flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Use NNDR to improve result
matches = matcher.knnMatch(des1, des2, k=2)
good = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

img6 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None,
                         flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


cv.imwrite("matches1.png", img5)
cv.imwrite("matches2.png", img6)
