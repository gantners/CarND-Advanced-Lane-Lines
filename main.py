import pickle
import matplotlib
import util
import os
from moviepy.editor import VideoFileClip

matplotlib.use('TkAgg')
import numpy as np

import matplotlib.pyplot as plt
import cv2
from Lane import Lane


def process_frame(frame, draw=False):
    global lane

    mtx = lane.calibration['mtx']
    dist = lane.calibration['dist']

    # go ahead with testimage and undistore it
    undist = util.undistort(frame, mtx, dist)

    # Apply each of the thresholding functions
    gradx = util.abs_sobel_thresh(undist, orient='x', thresh_min=12, thresh_max=100)
    grady = util.abs_sobel_thresh(undist, orient='y', thresh_min=25, thresh_max=100)
    mag_binary = util.mag_thresh(undist, sobel_kernel=9, mag_thresh=(20, 120))
    dir_binary = util.dir_threshold(undist, sobel_kernel=15, thresh=(0.2, 1.3))
    c_grad = util.color_and_gradient(undist)

    gray_select = cv2.cvtColor(undist, cv2.COLOR_RGB2GRAY)
    r_select, r_binary = util.color_select(undist, channel=0, thresh=(180, 255), color_space=None)
    h_select, h_binary = util.color_select(undist, channel=0, thresh=(15, 100), color_space=cv2.COLOR_RGB2HLS)
    l_select, l_binary = util.color_select(undist, channel=1, thresh=(180, 255), color_space=cv2.COLOR_RGB2HLS)
    s_select, s_binary = util.color_select(undist, channel=2, thresh=(90, 255), color_space=cv2.COLOR_RGB2HLS)
    v_select, v_binary = util.color_select(undist, channel=2, thresh=(50, 255), color_space=cv2.COLOR_RGB2HSV)

    hv_binary = util.hv_select(undist, s_thresh=(100, 255), h_tresh=(50, 255))

    combined = np.zeros_like(hv_binary)
    combined[((gradx == 1) & (grady == 1)) | (hv_binary == 1) | ((mag_binary == 1) & (dir_binary == 1)) | ((c_grad == 1) & (s_binary == 1))] = 1

    # finally warp image
    binary_warped, Minv = util.create_warped(combined, dynamic=True, draw=draw)

    # go for detecting lines on bird view
    # initial fit if not exist
    if not lane.detected:
        lane.fit(binary_warped, draw)

    result2, sliding_img = lane.fit_with_history(binary_warped, Minv, draw=draw)

    # Drawing the lines back down onto the road
    color_result = cv2.warpPerspective(sliding_img, Minv, (sliding_img.shape[1], sliding_img.shape[0]))
    color_sliding = cv2.addWeighted(undist, 1, color_result, 0.3, 0)

    # measure curvature
    lane.measuring_curvature()

    # draw lane
    result = lane.draw_lane(undist, sliding_img[:, :, 0], Minv, lane.recent_left_fit, lane.recent_right_fit,
                            draw_fake=False,
                            draw=False)
    if draw:
        images = [gray_select, r_select, r_binary, s_select, s_binary, h_select, h_binary, l_select, l_binary, v_select,
                  v_binary, hv_binary]
        titles = ['gray_select', 'r_select', 'r_binary', 's_select', 's_binary', 'h_select', 'h_binary', 'l_select',
                  'l_binary', 'v_select', 'v_binary', 'hv_binary']
        util.draw_grid(images, titles, 3, 4)

        images = [gradx, grady, mag_binary, dir_binary, c_grad, combined]
        titles = ['gradx', 'grady', 'mag_binary', 'dir_binary', 'c_grad', 'combined']
        util.draw_grid(images, titles, 2, 3)

    if draw:
        plt.imshow(result)
        plt.show()
    return result


def test_calibration(mtx, dist):
    for i in range(1, 7):
        img = cv2.imread('./test_images/test' + str(i) + '.jpg')
        undistorted = util.undistort(img, mtx, dist)
        filename = './output_images/test' + str(i) + '_undistorted.jpg'
        util.plot_and_save(img, undistorted, filename)


def test_lanes(mtx, dist):
    for i in range(1, 7):
        img = cv2.imread('./test_images/test' + str(i) + '.jpg')
        processed = process_frame(img)
        filename = './output_images/test' + str(i) + '_processed.jpg'
        util.plot_and_save(filename, img, processed, filename)


def test_calibration2(mtx, dist):
    for i in range(1, 20):
        img = cv2.imread('./camera_cal/calibration' + str(i) + '.jpg')
        undistorted = util.undistort(img, mtx, dist)
        filename = './output_images/calibration' + str(i) + '_undistorted.jpg'
        util.plot_and_save(filename, img, undistorted, filename)


def try1(cal):
    mtx = cal['mtx']
    dist = cal['dist']

    # go ahead with testimage and undistore it
    test_img = cv2.imread('./test_images/straight_lines2.jpg')
    undistorted_img = util.undistort(test_img, mtx, dist)
    # proceed from above undistorted image
    image = undistorted_img

    # plot_and_save(test_img,undistorted_img,'test_img.jpg')

    # image = undistorted_img
    # binary = rgb_select(image)
    # image = undistorted_img
    # binary_output = rgb_select(hls_select)


    # Choose a Sobel kernel size
    ksize = 5  # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = util.abs_sobel_thresh(image, orient='x', thresh_min=20, thresh_max=100)
    grady = util.abs_sobel_thresh(image, orient='y', thresh_min=20, thresh_max=100)
    mag_binary = util.mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 120))
    dir_binary = util.dir_threshold(image, sobel_kernel=ksize, thresh=(0, np.pi / 2))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    draw_dir_binary = True
    save_dir_binary = False
    if draw_dir_binary:
        # Plot the result
        f = plt.figure()

        ax0 = f.add_subplot(231)
        ax1 = f.add_subplot(232)
        ax2 = f.add_subplot(233)
        ax3 = f.add_subplot(234)
        ax4 = f.add_subplot(235)
        ax5 = f.add_subplot(236)

        f.tight_layout()

        ax0.imshow(image)
        ax0.set_title('Original', fontsize=10)
        ax1.imshow(gradx, cmap='gray')
        ax1.set_title('gradx', fontsize=10)
        ax2.imshow(grady, cmap='gray')
        ax2.set_title('grady', fontsize=10)
        ax3.imshow(mag_binary)
        ax3.set_title('mag_binary', fontsize=10)
        ax4.imshow(dir_binary)
        ax4.set_title('dir_binary', fontsize=10)
        ax5.imshow(combined)
        ax5.set_title('combined', fontsize=10)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        filename = './output_images/test1_dir_binary.jpg'
        if save_dir_binary:
            plt.savefig(filename)
        plt.show()
        plt.close()

    combined_binary = util.color_and_gradient(undistorted_img)

    binary_warped, Minv = util.create_warped(combined_binary)

    '''
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    '''

    lane = Lane()
    left_fit, right_fit = util.sliding_window_init(lane, binary_warped)
    binary_warped = util.sliding_window(lane, binary_warped, left_fit, right_fit, Minv)
    # Read in a thresholded image
    # util.alternate_sliding(binary_warped)
    lane.measuring_curvature()


def load_cal():
    with open('calibration_data.p', 'rb') as file:
        cal = pickle.load(file)
        ret = cal['ret']
        mtx = cal['mtx']
        dist = cal['dist']
        rvecs = cal['rvecs']
        tvecs = cal['tvecs']
    return cal


def process_video(input, output):
    clip1 = VideoFileClip(input)
    white_clip = clip1.fl_image(process_frame)
    white_clip.write_videofile(output, audio=False)


lane = Lane()


def run():
    cal_file = 'calibration_data.p'
    do_cal = True if not os.path.exists(cal_file) else False

    test_img = cv2.imread('./test_images/straight_lines1.jpg')

    if do_cal:
        print('Calibrating image distortion')
        # run calibration
        images, objpoints, imgpoints = util.read_data()

        ret, mtx, dist, rvecs, tvecs = util.calibrate(test_img, objpoints, imgpoints)
        cal = {}
        cal['ret'] = ret
        cal['mtx'] = mtx
        cal['dist'] = dist
        cal['rvecs'] = rvecs
        cal['tvecs'] = tvecs
        with open(cal_file, "wb") as file:
            pickle.dump(cal, file)

        lane.calibration = cal
        test_calibration(cal['mtx'], cal['dist'])
    else:
        print('Load calibration data...')
        lane.calibration = load_cal()

    # try1(lane.calibration)
    # test_calibration2(calibration['mtx'], calibration['dist'])
    # process_frame(cv2.imread('./test_images/test1.jpg'), True)
    process_video('project_video.mp4', 'project_test.mp4')
    # process_video('challenge_video.mp4', 'project_challenge.mp4')
    # process_video('harder_challenge_video.mp4', 'project_harder_challenge.mp4')
    # test_lanes(calibration['mtx'], calibration['dist'])


if __name__ == '__main__':
    run()
