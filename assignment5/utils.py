# Import numpy and OpenCV
import numpy as np
import cv2
import matplotlib.pyplot as plt


def moving_average(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size) / window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed


def smooth(path, radius):
    smoothed_trajectory = np.copy(path)
    # Filter the x, y and angle curves
    for j in range(3):
        smoothed_trajectory[:, j] = moving_average(path[:, j], radius=radius)

    return smoothed_trajectory


def fix_border(image, scale):
    s = image.shape
    # Scale the image without moving the center
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, scale)
    image = cv2.warpAffine(image, T, (s[1], s[0]))
    return image


def plot_trajectory(trajectory, title):
    time = range(trajectory.shape[0])
    plt.plot(time, trajectory[:, 0], label="dx")
    plt.plot(time, trajectory[:, 1], label="dy")
    plt.plot(time, trajectory[:, 2], label="da")
    plt.legend()
    plt.savefig(title)
    plt.clf()

