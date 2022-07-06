import cv2 as cv
from find_features import find_face_features
from prewrap import prewarp, prewarp_face, prewarp_manual
from morph import morph
import numpy as np
from postwarp import getPoints, homography_points

SIMPLE = True
PREWARP = False
POSTWARP = False

if __name__ == "__main__":
    # Read image from file
    image1 = cv.imread('images/einstein1.jpg')
    image2 = cv.imread('images/einstein3.jpg')

    gray1 = cv.imread('images/einstein1.jpg', 0)
    gray2 = cv.imread('images/einstein3.jpg', 0)

    h1, w1 = gray1.shape
    h2, w2 = gray2.shape

    if PREWARP:
        # Rectify both image
        H1, H2 = prewarp_face(gray1, gray2)
        new_size = int(np.sqrt(np.power(image1.shape[0], 2) + np.power(image1.shape[1], 2)))
        image1 = cv.warpPerspective(image1, H1, (new_size, new_size))
        image2 = cv.warpPerspective(image2, H2, (new_size, new_size))
        cv.imwrite("images/results/einstein/einstein_prewarp1_face.png", image1)
        cv.imwrite("images/results/einstein/einstein_prewarp2_face.png", image2)

        gray1 = cv.warpPerspective(gray1, H1, (w1, h1))
        gray2 = cv.warpPerspective(gray2, H2, (w1, h1))

    # Apply an image morph to obtain an intermediary view
    pts1, pts2 = find_face_features(gray1, gray2)

    img = morph(image1, image2, pts1, pts2, 0.5)
    cv.imwrite("images/results/einstein/einstein_morph_face.jpg", img)

    if POSTWARP:
        # Postwarp. Select point correspondences manually using mask, find homography and morph
        m_points = getPoints(img)
        im = cv.imread('images/mask_1.jpg')
        p_points = getPoints(np.array(im).astype(np.uint8))

        # Find homography using the points
        H_s = homography_points(m_points, p_points)
        h, w, _ = image1.shape

        # warp image to desired plane
        final_morph = cv.warpPerspective(img, H_s, (h1, w1))
        cv.imwrite("images/results/einstein/einstein_postwarp_face.jpg", final_morph)
