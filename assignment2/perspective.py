# importing the module
import cv2
import numpy as np

img = cv2.imread('b.jpg', 1)
pts1 = np.float32([[118, 209], [497, 45], [86, 604], [548, 520]])
pts2 = np.float32([[100, 200], [500, 200], [100, 600], [500, 600]])
M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img, M, (720, 1080))
cv2.imshow('output', dst)
cv2.imwrite('output.jpg', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
