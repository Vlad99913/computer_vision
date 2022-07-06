import cv2 as cv
import numpy as np
from find_features import find_face_features


def get_triangles(img, pts):
    """
    Get Delauny triangulation
    :param img: input image
    :param pts: feature points
    :return: Delauny triangulation of the space
    """
    # Rectangle to be used with Subdiv2D
    size = img.shape
    rect = (0, 0, size[1], size[0])

    # Create an instance of Subdiv2D
    subdiv = cv.Subdiv2D(rect)

    # Insert points into subdiv
    for x, y in pts:
        subdiv.insert((int(x), int(y)))

    # Obtain Delauny triangulation of the image
    triangles = subdiv.getTriangleList()
    return triangles


def list_to_triangles(pt_list):
    """
    Convert the lists of 6 floats output from Subdiv2D.getTriangleList() into lists with 3 integer point coordinates
    :param pt_list: output from Subdiv2D.getTriangleList()
    :return: lists with 3 integer point coordinates
    """
    triangles = []
    for t in pt_list:
        pt1 = [int(t[0]), int(t[1])]
        pt2 = [int(t[2]), int(t[3])]
        pt3 = [int(t[4]), int(t[5])]
        triangle = [pt1, pt2, pt3]
        triangles.append(triangle)

    return triangles


def map_delaunay(triangles_A, points_A, points_B, points_C):
    """
    Using delauny triangle map for one image, get triangles for the two other images as well
    :param triangles_A: triangle map in the 1st image
    :param points_A: feature points in the 1st image
    :param points_B: feature points in the 2nd image
    :param points_C: feature points in the 3rd image
    :return: Delauny triangle maps for the 2nd and 3rd images
    """
    triangles_B = []
    triangles_C = []

    for tri in triangles_A:
        tri_B = []
        tri_C = []
        for i in range(0, 6, 2):
            index = np.where(points_A == tri[i])
            for idx in index[0]:
                if points_A[idx][1] == tri[1 + i]:
                    tri_B.extend(points_B[idx])
                    tri_C.extend(points_C[idx])

        triangles_B.append(tri_B)
        triangles_C.append(tri_C)

    return triangles_B, triangles_C


def apply_affine_transform(src, src_triangle, dst_triangle, size):
    # Given a pair of triangles, find the affine transform
    warp_mat = cv.getAffineTransform(src_triangle, dst_triangle)

    # Apply the affine transform just found to the image
    dst = cv.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT101)

    return dst


def morph_triangle(img1, img2, img, t1, t2, t, alpha):
    """
    Blend corresponding triangular regions in 2 image
    :param img1: input image
    :param img2: input image
    :param img: output image
    :param t1: trianle in img1
    :param t2: triangle in img2
    :param t: triangle in output image
    :param alpha: blending parameter (float between 0.0 and 1.0)
    """
    # Find bounding rectangle for each triangle
    r1 = cv.boundingRect(np.float32([t1]))
    r2 = cv.boundingRect(np.float32([t2]))
    r = cv.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1_rect = []
    t2_rect = []
    t_rect = []

    for i in range(0, 3):
        t_rect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv.fillConvexPoly(mask, np.int32(t_rect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2_rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warp_image1 = apply_affine_transform(img1_rect, np.float32(t1_rect), np.float32(t_rect), size)
    warp_image2 = apply_affine_transform(img2_rect, np.float32(t2_rect), np.float32(t_rect), size)

    # Alpha blend rectangular patches
    img_rect = (1.0 - alpha) * warp_image1 + alpha * warp_image2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + img_rect * mask


def morph(img1, img2, pts1, pts2, alpha):
    """
    Use Delauny triangulation to blend 2 images
    :param img1: input image
    :param img2: input image
    :param pts1: feature points in the first image
    :param pts2: feature points
    :param alpha: float between 0 and 1, the closer it is to 1, the more the result resembles img2
    :return: blended result
    """
    # Get intermediate points for generated image
    pts_morph = (1 - alpha) * pts1 + alpha * pts2

    # Allocate space for final output
    img_morph = np.zeros(img1.shape, dtype=img1.dtype)

    # Apply Delaunay triangulation on both images images
    t_list1 = get_triangles(img1, pts1)
    t_list2, t_list = map_delaunay(t_list1, pts1, pts2, pts_morph)

    t_list1 = list_to_triangles(t_list1)
    t_list2 = list_to_triangles(t_list2)
    t_list = list_to_triangles(t_list)

    # Blend generated triangular regions
    for t in range(len(t_list1)):
        morph_triangle(img1, img2, img_morph, t_list1[t], t_list2[t], t_list[t], alpha)

    return img_morph


if __name__ == "__main__":
    img1 = cv.imread("images/donald_trump.jpg")
    img2 = cv.imread("images/hillary_clinton.jpg")

    gray1 = cv.imread("images/donald_trump.jpg", 0)
    gray2 = cv.imread("images/hillary_clinton.jpg", 0)

    pts1, pts2 = find_face_features(gray1, gray2)

    img = morph(img1, img2, pts1, pts2, 0.4)

    cv.imwrite("images/morph3.jpg", img)
