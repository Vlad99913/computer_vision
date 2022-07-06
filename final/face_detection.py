from imutils import face_utils
import dlib
import cv2 as cv

path = "shape_predictor_68_face_landmarks.dat"

img = cv.imread("images/dt.jpg")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# get faces in image
rect = detector(gray, 0)

# for each detected face, find the landmarks
for i, face in enumerate(rect):
    shape = predictor(gray, face)
    shape = face_utils.shape_to_np(shape)

    # Draw on our image, all the found coordinate points (x,y)
    counter = 0
    for (x, y) in shape:
        counter += 1
        cv.circle(img, (x, y), 2, (0, 255, 0), -1)

    print(counter)
    # write the result
    cv.imwrite('face_features{}.png'.format(i), img)
