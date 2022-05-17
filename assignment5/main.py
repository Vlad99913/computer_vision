# Import numpy and OpenCV
import numpy as np
import cv2


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


# Read input video
cap = cv2.VideoCapture('video.mp4')

# Get frame count and frame rate
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Get width and height of video stream
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec for output video
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

# Set up output video
out = cv2.VideoWriter('video_out.mp4', fourcc, fps, (w, h))

# Read first frame
_, prev = cap.read()

# Convert frame to grayscale
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

# Pre-define transformation-store array
transforms = np.zeros((n_frames - 1, 3), np.float32)

for i in range(n_frames - 2):
    # Detect feature points in previous frame
    prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                       maxCorners=200,
                                       qualityLevel=0.01,
                                       minDistance=30,
                                       blockSize=3)

    # Read next frame
    success, curr = cap.read()
    if not success:
        break

    # Convert to grayscale
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow (i.e. track feature points)
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

    # Sanity check
    assert prev_pts.shape == curr_pts.shape

    # Filter only valid points
    idx = np.where(status == 1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]

    # Find transformation matrix
    m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)

    # Extract translation
    dx = m[0, 2]
    dy = m[1, 2]

    # Extract rotation angle
    da = np.arctan2(m[1, 0], m[0, 0])

    # Store transformation
    transforms[i] = [dx, dy, da]

    # Move to next frame
    prev_gray = curr_gray

# Compute trajectory using cumulative sum of transformations
trajectory = np.cumsum(transforms, axis=0)

# Smooth the trajectory
smooth_trajectory = smooth(trajectory, 30)

# Calculate difference in smoothed_trajectory and trajectory
difference = smooth_trajectory - trajectory

# Calculate newer transformation array
transforms_smooth = transforms + difference

# Reset stream to first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Write n_frames-1 transformed frames
for i in range(n_frames - 2):
    # Read next frame
    success, frame = cap.read()
    if not success:
        break

    # Extract transformations from the new transformation array
    dx = transforms_smooth[i, 0]
    dy = transforms_smooth[i, 1]
    da = transforms_smooth[i, 2]

    # Reconstruct transformation matrix accordingly to new values
    m = np.zeros((2, 3), np.float32)
    m[0, 0] = np.cos(da)
    m[0, 1] = -np.sin(da)
    m[1, 0] = np.sin(da)
    m[1, 1] = np.cos(da)
    m[0, 2] = dx
    m[1, 2] = dy

    # Apply affine wrapping to the given frame
    frame_stabilized = cv2.warpAffine(frame, m, (w, h))

    # Fix border artifacts
    frame_stabilized = fix_border(frame_stabilized, 1.2)

    # Write the frame to the file
    out.write(frame_stabilized)

cap.release()
out.release()


