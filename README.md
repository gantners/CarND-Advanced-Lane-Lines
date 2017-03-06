**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
[image0]: ./output_images/Calibration.PNG "Calibration"
[image1]: ./output_images/test1_undistorted.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/grid3.png "Grid"
[image4]: ./output_images/combined3.png "Combined"
[image5]: ./output_images/sliding_window2.png "Fit Visual"
[image6]: ./output_images/successful.png "Output"
[image7]: ./output_images/warped_color.png "Warped Image"
[image8]: ./output_images/towarp_roi.png "Image with ROI"
[image9]: ./output_images/warped_roi.png "Warped Image with ROI"
[image10]: ./output_images/calibration1_undistorted.jpg "Undistorted calibration"
[image11]: ./output_images/binary_warped.png "Binary warped image"
[image12]: ./output_images/histo.PNG "Histogram"
[image13]: ./output_images/radius.PNG "Histogram"
[image14]: ./output_images/fakedata.PNG "Fake data"

[video1]: ./project_video.mp4 "Original Video"
[video2]: ./project_test.mp4 "Test Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in `util.py`. 

Reading calibration data: line 29-59

`

    def read_data (calibration_files='./camera_cal/calibration*.jpg', draw=False):
    #prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
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
`

Calibrate camera:

`
def calibrate(img, objpoints, imgpoints):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    `

Undistorting image:

`
def undistort(img, mtx, dist, draw=False):
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    if draw:
        plot_and_save(img, undistorted)
    return undistorted
`

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  
Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.
`imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

![alt text][image0]

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image10]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image1]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image, see also grid image:
`combined[((gradx == 1) & (grady == 1)) | (hv_binary == 1) | ((mag_binary == 1) & (dir_binary == 1)) | ((c_grad == 1) & (s_binary == 1))] = 1`

(thresholding steps at lines 24 through 44 in `main.py`).  

Color channel selection:

![alt text][image3]

Treshold selection:

![alt text][image4]


####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `create_warped()`, which appears in lines 300 through 359 in the file `util.py` 
The src and destination points where calculated dynamically based on the input image shape. I chose a wider area instead of straight in top
of the lane lines to be able to detect sharp curves better.

```
def create_warped(to_warp, dynamic=False, draw=False):
    width = to_warp.shape[1]
    height = to_warp.shape[0]
    xcenter = width / 2
    hcenter = height / 2
    space_top = 100
    space_bottom_side = 100
    space_bottom = 50
    hcorrection = 100

    src = np.float32([[680, 445], [1045, 675], [250, 675], [600, 445]])
    dst = np.float32([[1045, 0], [1045, 675], [250, 675], [250, 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 540, 460      | 100, 0        | 
| 100, 670      | 100, 670      |
| 1180, 670     | 1180, 670     |
| 740, 460      | 1180, 0       |


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image8]

![alt text][image9]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

This code was mostly taken from the lectures. First a histogram of the second lower half of the image was taken, which looks similar to this (`util.py` lines 446 through 464):

![alt text][image12]

As the input was a binary warped image which has only values 0 or 1 it gave a good starting point to do a sliding window search (`util.py` lines 466 through 530):

![alt text][image5]


If we once initially detected the lines, we do not need to do a blind search again, instead we can search within a margin (i choose 100px) around the 
left and right fittings previously detected (`util.py` lines 556 through 634)
In case we could not detect the lines anymore, a basic fallback to the initial blind search is activated. (`util.py` lines 582 through 592)

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The lane curvature was measured as suggested in the lectures by drawing fake points around the detected points to have some more data to finally
get some polynomial fits which is able to draw our curves:

![alt text][image13]

I did this in lines 65 through 111 in my code in `Lane.py`

![alt text][image14]



####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 113 through 185 in my code in `Lane.py` in the function `draw_lane()`.  
Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_test.mp4)



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

As i was already late on time, i chose to stick mostly with the parts of the lectures. From the testing it seems the key to success on the challenge is 
the right color and thresholding selection, detecting sharp curves and adding robustness by doing sanity checks like line parallelism or same slopes.
I also generated the challenge videos but failed fast on dark shadows or sharp curves.
