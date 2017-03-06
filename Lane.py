import matplotlib

import util

matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import cv2


class Lane:
    def __init__(self):
        # image calibration data
        self.calibration = None

        # was the line detected in the last iteration?
        self.detected = False

        # x values of the last n fits of the line
        self.recent_xfitted = []

        # average x values of the fitted line over the last n iterations
        self.bestx = None

        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None

        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]

        # radius of curvature of the line in some units
        self.radius_of_curvature = None

        # distance in meters of vehicle center from the line
        self.line_base_pos = None

        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')

        # x values for detected line pixels
        self.allx = None

        # y values for detected line pixels
        self.ally = None

        self.recent_left_fit = None
        self.left_coeff = 3e-4

        self.recent_right_fit = None
        self.right_coeff = 3e-4

        self.left_curverad = None
        self.right_curverad = None

    def fit(self, binary_warped,draw=False):
        return util.sliding_window_init(self, binary_warped, draw=draw)

    def fit_with_history(self, binary_warped, Minv, draw=True):
        return util.sliding_window(self, binary_warped, self.recent_left_fit, self.recent_right_fit, Minv, draw=draw)

    def base_pos(self):
        pass

    # measuring and set curvature
    def measuring_curvature(self):
        # Generate some fake data to represent lane-line pixels
        ploty = np.linspace(0, 719, num=720)  # to cover same y-range as image
        # calculate arbitrary quadratic coefficient
        # For each y position generate random x position within +/-50 pix
        # of the line base position in each case (x=200 for left, and x=900 for right)
        leftx = np.array([200 + (y ** 2) * self.left_coeff + np.random.randint(-50, high=51)
                          for y in ploty])
        rightx = np.array([900 + (y ** 2) * self.right_coeff + np.random.randint(-50, high=51)
                           for y in ploty])

        leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
        rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

        # Fit a second order polynomial to pixel positions in each fake lane line
        left_fit = np.polyfit(ploty, leftx, 2)
        right_fit = np.polyfit(ploty, rightx, 2)

        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)
        left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
        right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit[0])
        # print(left_curverad, right_curverad)
        # Example values: 1926.74 1908.48

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (
            2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (
            2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])
        # Now our radius of curvature is in meters
        # print(left_curverad, 'm', right_curverad, 'm')
        # Example values: 632.1 m    626.2 m
        self.left_curverad = left_curverad
        self.right_curverad = right_curverad
        return left_curverad, right_curverad

    def draw_lane(self, undist, warped, Minv, left_fit, right_fit, draw_fake=False, draw=False):
        # Generate some fake data to represent lane-line pixels
        ploty = np.linspace(0, 719, num=720)  # to cover same y-range as image
        # Drawing
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Generate some fake data to represent lane-line pixels
        ploty = np.linspace(0, 719, num=720)  # to cover same y-range as image
        quadratic_coeff = 3e-4  # arbitrary quadratic coefficient
        # For each y position generate random x position within +/-50 pix
        # of the line base position in each case (x=200 for left, and x=900 for right)
        leftx = np.array([200 + (y ** 2) * quadratic_coeff + np.random.randint(-50, high=51)
                          for y in ploty])
        rightx = np.array([900 + (y ** 2) * quadratic_coeff + np.random.randint(-50, high=51)
                           for y in ploty])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
        rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Plot up the fake data
        mark_size = 3

        if draw_fake:
            plt.plot(leftx, ploty, 'o', color='red', markersize=mark_size)
            plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)
            plt.plot(left_fitx, ploty, color='green', linewidth=3)
            plt.plot(right_fitx, ploty, color='green', linewidth=3)
            plt.xlim(0, 1280)
            plt.ylim(0, 720)
        plt.gca().invert_yaxis()  # to visualize as we do the images

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        camera_center = (left_fitx[-1] + right_fitx[-1]) / 2
        center_diff = (camera_center - warped.shape[1] / 2) * xm_per_pix

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        # newwarp = create_warped(color_warp)
        newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        color = (200, 200, 200)
        thickness = 2
        font_scale = 2.0
        cv2.putText(result, 'Left curve:  ' + str(round(self.left_curverad, 2)) + '(m)', (50, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color, thickness)
        cv2.putText(result, 'Right curve: ' + str(round(self.right_curverad, 2)) + '(m)', (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color, thickness)
        cv2.putText(result, 'Off from center: ' + str(round(center_diff, 2)) + 'm', (50, 225), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color, thickness)

        if draw:
            plt.imshow(result)
            plt.show()
        plt.close()
        return result
