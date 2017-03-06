import pickle
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.image as mpimg
import numpy as np

import matplotlib.pyplot as plt
import cv2
import glob


def plot_and_save(title, raw_image, undistorted, filename=None, cmap=None):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 9))
    f.tight_layout()
    ax1.imshow(raw_image, cmap=cmap)
    ax1.set_title('Original Image', fontsize=10)
    ax2.imshow(undistorted, cmap=cmap)
    ax2.set_title('Modified Image', fontsize=10)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    f.suptitle(title, fontsize=14)

    if filename is not None:
        plt.savefig(filename)
    plt.show()
    plt.close()


def read_data(calibration_files='./camera_cal/calibration*.jpg', draw=False):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(calibration_files)

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            if draw:
                img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(1500)
    return images, objpoints, imgpoints


# performs image distortion correction and returns the undistorted image
def undistort(img, mtx, dist, draw=False):
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    if draw:
        plot_and_save(img, undistorted)
    return undistorted


# performs the camera calibration
def calibrate(img, objpoints, imgpoints):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


def perspective_transform(img, src, dst, reverse=False):
    img_size = (img.shape[1], img.shape[0])

    # Compute the perspective transform, M, given source and destination points:
    M = cv2.getPerspectiveTransform(src, dst)

    if reverse:
        # Compute  the  inverse perspective transform:
        warped = cv2.getPerspectiveTransform(dst, src)
    else:
        # Warp an image using the perspective transform, M:
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped


# Define a function that takes an image, number of x and y points,
# camera matrix and distortion coefficients
def corners_unwarp(img, nx, ny, mtx, dist, draw=False):
    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Convert undistorted image to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    # Search for corners in the grayscaled image
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret:
        # If we found corners, draw them! (just for fun)
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        # Choose offset from image corners to plot detected corners
        # This should be chosen to present the result at the proper aspect ratio
        # My choice of 100 pixels is not exact, but close enough for our purpose here
        offset = 100  # offset for dst points
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])

        # For source points I'm grabbing the outer four detected corners
        src = np.float32([corners[0], corners[nx - 1], corners[-1], corners[-nx]])
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result
        # again, not exact, but close enough for our purposes
        dst = np.float32([[offset, offset], [img_size[0] - offset, offset],
                          [img_size[0] - offset, img_size[1] - offset],
                          [offset, img_size[1] - offset]])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, M, img_size)

    # Return the resulting image and matrix
    return warped, M


# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output


# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel)
    # Calculate the gradient magnitude
    abs_sobelxy = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scaled_sobel = np.uint8(255 * abs_sobelxy / np.max(abs_sobelxy))
    thresh_min = mag_thresh[0]
    thresh_max = mag_thresh[1]
    # Create a binary image of ones where threshold is met, zeros otherwise
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return sxbinary


# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    direction = np.arctan2(abs_sobely, abs_sobelx)
    # scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    sxbinary = np.zeros_like(direction)
    sxbinary[(direction >= thresh_min) & (direction <= thresh_max)] = 1
    return sxbinary

# draw a grid of iamges with titles
def draw_grid(images, titles, rows, cols):
    w = images[0].shape[1]
    h = images[0].shape[0]
    fig, axes = plt.subplots(rows, cols, subplot_kw={'xticks': [], 'yticks': []})
    for ax, img, title in zip(axes.flat, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
    plt.tight_layout()
    plt.subplots_adjust(left=0., right=1., top=0.9, bottom=0.)
    fig.suptitle('Grid', fontsize=14)
    plt.show()
    #filename = './output-images/pipeline.jpg'
    # if filename is not None:
    # plt.savefig(filename)


def color_select(image, channel=0, thresh=(200, 255), color_space=None):
    """
    The R channel does a reasonable job of highlighting the lines, and you can apply a similar threshold to find lane-line pixels:
    The S channel picks up the lines well, so let's try applying a threshold there:
    You can also see that in the H channel, the lane lines appear dark
    :param image:
    :param channel:
    :param thresh:
    :param color_space:
    :return:
    """
    if color_space is not None:
        img = cv2.cvtColor(image, color_space)
    else:
        img = image
    space = img[:, :, channel]
    binary_color = np.zeros_like(space)
    binary_color[(space > thresh[0]) & (space <= thresh[1])] = 1
    return space,binary_color


# Define a function that thresholds the S-channel of HLS
def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output


# Define a function that thresholds the H-channel of HSV
def hsv_select(img, thresh=(0, 255)):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h_channel = hsv[:, :, 2]
    v_binary = np.zeros_like(h_channel)
    v_binary[(h_channel >= thresh[0]) & (h_channel <= thresh[1])] = 1
    return v_binary


def hv_select(image, s_thresh, h_tresh):
    h_binary = hsv_select(image, h_tresh)
    s_binary = hls_select(image, s_thresh)
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(h_binary)
    combined_binary[(h_binary == 1) & (s_binary == 1)] = 1
    return combined_binary


def color_and_gradient(img, draw=False):
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Threshold color channel
    s_thresh_min = 50
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    # Plotting thresholded images
    if draw:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.set_title('Stacked thresholds')
        ax1.imshow(color_binary)

        ax2.set_title('Combined S channel and gradient thresholds')
        ax2.imshow(combined_binary, cmap='gray')
        plt.show()
        plt.close()
    return combined_binary


def create_warped(to_warp, dynamic=False, draw=False):
    # source 1280,720
    width = to_warp.shape[1]
    height = to_warp.shape[0]
    xcenter = width / 2
    hcenter = height / 2
    space_top = 100
    space_bottom_side = 100
    space_bottom = 50
    hcorrection = 100

    # dynamic calculated
    if dynamic:
        x1, y1 = space_bottom_side, height - space_bottom
        x2, y2 = xcenter - space_top, hcenter + hcorrection
        x3, y3 = xcenter + space_top, hcenter + hcorrection
        x4, y4 = width - space_bottom_side, height - space_bottom
        src = np.float32([[x3, y3], [x4, y4], [x1, y1], [x2, y2]])
        dst = np.float32([[x4, 0], [x4, y4], [x1, y1], [x1, 0]])
        dsrc = np.int32([[x3, y3], [x4, y4], [x1, y1], [x2, y2]])
        ddst = np.int32([[x4, 0], [x4, y4], [x1, y1], [x1, 0]])
        #print(src)
        #print(dst)
    else:
        # fixed
        src = np.float32([[680, 445], [1045, 675], [250, 675], [600, 445]])
        dst = np.float32([[1045, 0], [1045, 675], [250, 675], [250, 0]])
        dsrc = np.int32([[680, 445], [1045, 675], [250, 675], [600, 445]])
        ddst = np.int32([[1045, 0], [1045, 675], [250, 675], [250, 0]])

    '''
    # rect = to_warp.copy()
    # cv2.polylines(rect, [dsrc], True, (0, 255, 255), thickness=2)
    # cv2.polylines(rect, [ddst], True, (255, 0, 255), thickness=2)
    # f, (ax1) = plt.subplots(1, 1, figsize=(6, 10))
    # ax1.set_title('binary')
    # ax1.imshow
    # plt.show()
    # plt.close()

    img = cv2.imread('./test_images/hard.jpg')
    cv2.polylines(img, [dsrc], True, (0, 255, 255), thickness=2)
    plt.imshow(img)
    plt.show()
    '''

    # Given src and dst points, calculate the perspective transform matrix
    Minv = M = cv2.getPerspectiveTransform(dst, src)
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(to_warp, M, (width, height), cv2.INTER_LINEAR)

    if draw:
        f, (ax1) = plt.subplots(1, 1, figsize=(6, 10))
        ax1.set_title('warped')
        ax1.imshow(warped, cmap='gray')
        plt.show()
        plt.close()

    return warped, Minv


def pipeline(img_p, s_thresh=(170, 255), sx_thresh=(20, 200)):
    image_orig = np.copy(img_p)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(image_orig, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:, :, 1]
    s_channel = hsv[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
    return color_binary


def equalizeHist_gray(img):
    """
    Apply clahe and histogram equalization to an image for gray channel
    :param img: image to equalize
    :return:
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
    cl = clahe.apply(gray)
    hist = cv2.equalizeHist(cl)
    return hist[:, :, np.newaxis]


def equalizeHist_color(img):
    """
    Apply clahe and histogram equalization to an image for each channel
    :param img: image to equalize
    :return:
    """
    image = np.empty(img.shape)
    for c in range(img.shape[2]):
        channel = img[:, :, c]
        channel = channel.astype(np.uint8)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
        channel = clahe.apply(channel)

        # http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
        channel = cv2.equalizeHist(channel)
        try:
            image[:, :, c] = channel
        except Exception as e:
            print(str(e))
    return image


def histogram(pipe_img, draw):
    color_binary = hv_select(pipe_img, s_thresh=(100, 255), h_tresh=(50, 255))
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(pipe_img)
    ax1.set_title('Original Image', fontsize=40)

    ax2.imshow(color_binary)
    ax2.set_title('Pipeline Result', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    histogram = np.sum(color_binary[color_binary.shape[0] / 2:, :], axis=0)

    if draw:
        plt.plot(histogram)
        plt.show()
        plt.close()
    return histogram


def sliding_window_init(lane, binary_warped, draw=False):
    """
    We search for the first time for lines to get a clue where the lane actually is
    :param binary_warped:
    :param draw:
    :return:
    """

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram1 = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram1.shape[0] / 2)
    leftx_base = np.argmax(histogram1[:midpoint])
    rightx_base = np.argmax(histogram1[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    if len(leftx) == 0:
        left_fit = lane.recent_left_fit
    else:
        left_fit = np.polyfit(lefty, leftx, 2)

    if len(rightx) == 0:
        right_fit = lane.recent_right_fit
    else:
        right_fit = np.polyfit(righty, rightx, 2)

    if draw:
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()
        plt.close()

    lane.recent_left_fit = left_fit
    lane.recent_right_fit = right_fit
    lane.detected = True

    return left_fit, right_fit


def sliding_window(lane, binary_warped, left_fit, right_fit, Minv, draw=False):
    """
    We already have a left and right fit from blind search, so we just search within margin windows
    :param binary_warped:
    :param left_fit:
    :param right_fit:
    :param Minv:
    :param draw:
    :return:
    """
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
        nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = (
        (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
            nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    if len(leftx) == 0:
        left_fit = lane.recent_left_fit
        lane.detected = False
    else:
        left_fit = np.polyfit(lefty, leftx, 2)

    if len(rightx) == 0:
        right_fit = lane.recent_right_fit
        lane.detected = False
    else:
        right_fit = np.polyfit(righty, rightx, 2)


    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # if draw:
    # Create an image to draw on an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))

    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # curves
    # newwarp = cv2.warpPerspective(result, Minv, (result.shape[1], result.shape[0]))
    # plt.imshow(newwarp)
    # plt.plot(left_fitx, ploty, color='blue')
    # plt.plot(right_fitx, ploty, color='red')
    # plt.show()

    lane.recent_left_fit = left_fit
    lane.recent_right_fit = right_fit
    lane.detected = True

    return result, window_img


# alternative sliding window approach
def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output


def find_window_centroids(warped, window_width, window_height, margin):
    window_centroids = []  # Store the (left,right) window centroid positions per level
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, :int(warped.shape[1] / 2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
    r_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, int(warped.shape[1] / 2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(warped.shape[1] / 2)

    # Add what we found for the first layer
    window_centroids.append((l_center, r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1, (int)(warped.shape[0] / window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(
            warped[int(warped.shape[0] - (level + 1) * window_height):int(warped.shape[0] - level * window_height), :],
            axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width / 2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, warped.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, warped.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
        # Add what we found for that layer
        window_centroids.append((l_center, r_center))

    return window_centroids


def alternate_sliding(warped, draw=False):
    # window settings
    window_width = 50
    window_height = 80  # Break image into 9 vertical layers since image height is 720
    margin = 100  # How much to slide left and right for searching
    window_centroids = find_window_centroids(warped, window_width, window_height, margin)

    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255

        # Draw the results
        template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
        zero_channel = np.zeros_like(template)  # create a zero color channle
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
        warpage = np.array(cv2.merge((warped, warped, warped)),
                           np.uint8)  # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # overlay the orignal road image with window results

    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped, warped, warped)), np.uint8)

    # Display the final results
    if draw:
        plt.imshow(output)
        plt.title('window fitting results')
        plt.show()

    return output
