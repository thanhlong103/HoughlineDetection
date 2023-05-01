import cv2
import numpy as np


# Function to calculate average.
def cal_avg(values):
    """Calculate average value."""
    if not (type(values) == 'NoneType'):
        if len(values) > 0:
            n = len(values)
        else:
            n = 1
        return sum(values) / n


# Function to draw lines.
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """Utility for drawing lines."""
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)


# Function to separate left-right lines.
def separate_left_right_lines(lines):
    """ Separate left and right lines depending on the slope. """
    left_lines = []
    right_lines = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if y1 > y2:  # Negative slope = left lane.
                    left_lines.append([x1, y1, x2, y2])
                elif y1 < y2:  # Positive slope = right lane.
                    right_lines.append([x1, y1, x2, y2])
    return left_lines, right_lines


# Function to extrapolate lines.
def extrapolate_lines(lines, upper_border, lower_border):
    """Extrapolate lines keeping in mind the lower and upper border intersections."""
    slopes = []
    consts = []

    if lines is not None:
        for x1, y1, x2, y2 in lines:
            slope = (y1 - y2) / (x1 - x2)
            slopes.append(slope)
            c = y1 - slope * x1
            consts.append(c)
    avg_slope = cal_avg(slopes)
    avg_consts = cal_avg(consts)

    # Calculate average intersection at lower_border.
    x_lane_lower_point = int((lower_border - avg_consts) / avg_slope)

    # Calculate average intersection at upper_border.
    x_lane_upper_point = int((upper_border - avg_consts) / avg_slope)

    return [x_lane_lower_point, lower_border, x_lane_upper_point, upper_border]


if __name__ == "__main__":

    # Reading the image.
    img = cv2.imread('./test_img1.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Convert to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Use global threshold based on grayscale intensity.
    threshold = cv2.inRange(gray, 150, 255)

    # Display images.
    cv2.imshow('Grayscale', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('Threshold', threshold)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Region masking: Select vertices according to the input image.
    roi_vertices = np.array([[[100, 540],
                              [900, 540],
                              [515, 320],
                              [450, 320]]])

    # Defining a blank mask.
    mask = np.zeros_like(threshold)

    # Defining a 3 channel or 1 channel color to fill the mask.
    if len(threshold.shape) > 2:
        channel_count = threshold.shape[2]  # 3 or 4 depending on the image.
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # Filling pixels inside the polygon.
    cv2.fillPoly(mask, roi_vertices, ignore_mask_color)

    # Constructing the region of interest based on where mask pixels are nonzero.
    roi = cv2.bitwise_and(threshold, mask)

    cv2.imshow('Initial threshold', threshold)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('Polyfill mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('Isolated roi', roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Perform Edge Detection.
    low_threshold = 50
    high_threshold = 100
    edges = cv2.Canny(roi, low_threshold, high_threshold)

    # Smooth with a Gaussian blur.
    kernel_size = 3
    canny_blur = cv2.GaussianBlur(edges, (kernel_size, kernel_size), 0)

    # Display images.
    cv2.imshow('Edge detection', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('Blurred edges', canny_blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Hough transform parameters set according to the input image.
    rho = 1
    theta = np.pi / 180
    threshold = 50
    min_line_len = 10
    max_line_gap = 20

    lines = cv2.HoughLinesP(canny_blur, rho, theta, threshold,
                            minLineLength=min_line_len, maxLineGap=max_line_gap)

    # Draw all lines found onto a new image.
    hough = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(hough, lines)

    print("Found {} lines, including: {}".format(len(lines), lines[0]))
    cv2.imshow('Hough', hough)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Define bounds of the region of interest.
    roi_upper_border = 340
    roi_lower_border = 540

    # Create a blank array to contain the (colorized) results.
    lanes_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    # Use above defined function to identify lists of left-sided and right-sided lines.
    lines_left, lines_right = separate_left_right_lines(lines)

    # Use above defined function to extrapolate the lists of lines into recognized lanes.
    lane_left = extrapolate_lines(lines_left, roi_upper_border, roi_lower_border)
    lane_right = extrapolate_lines(lines_right, roi_upper_border, roi_lower_border)
    draw_lines(lanes_img, [[lane_left]], thickness=10)
    draw_lines(lanes_img, [[lane_right]], thickness=10)

    # Display results.
    # Following step is optional and only used in the script for display convenience.
    hough1 = cv2.resize(hough, None, fx=0.5, fy=0.5)
    lanes_img1 = cv2.resize(lanes_img, None, fx=0.5, fy=0.5)
    comparison = cv2.hconcat([hough1, lanes_img1])
    cv2.imshow('Before and after extrapolation', comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    alpha = 0.8
    beta = 1.0
    gamma = 0.0
    image_annotated = cv2.addWeighted(img, alpha, lanes_img, beta, gamma)

    # Display the results, and save image to file.
    cv2.imshow('Annotated Image', image_annotated)
    cv2.imwrite('./Lane1-image.jpg', image_annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
