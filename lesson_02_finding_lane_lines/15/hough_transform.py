import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

plt.ion()

# Read in and grayscale the image
image = mpimg.imread('exit-ramp.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

# Define a kernel size and apply Gaussian smoothing
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

# Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
# plt.imshow(edges, cmap='Greys_r')
# plt.show()

# Next we'll create a masked edges image using cv2.fillPoly()
mask = np.zeros_like(edges)
ignore_mask_color = 255
# plt.imshow(mask, cmap='Greys_r')
# plt.show()

# This time we are defining a four sided polygon to mask
imshape = image.shape
top_hor_dist=460
bot_hor_dist=50
ver_dist=290
vertices = np.array([[(bot_hor_dist,imshape[0]),
                      (top_hor_dist, ver_dist),
                      (imshape[1] - top_hor_dist, ver_dist),
                      (imshape[1] - bot_hor_dist,imshape[0])]], dtype=np.int32)
cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)
# plt.imshow(masked_edges, cmap='Greys_r')
# plt.show()

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
# for threshold in range(10):
#     for min_line_length in range(5, 15):
#         for max_line_gap in range(1, 10):
threshold = 10     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 30 #minimum number of pixels making up a line
max_line_gap = 20    # maximum gap in pixels between connectable line segments
line_image = np.copy(image)*0 # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

# Iterate over the output "lines" and draw lines on a blank image
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

# Create a "color" binary image to combine with line image
color_edges = np.dstack((edges, edges, edges))

# Draw the lines on the edge image
lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
plt.imshow(lines_edges)
plt.title("%s - %s - %s" % (threshold, min_line_length, max_line_gap))
plt.show()

# solution Udacity:
# Here's how I did it: I went with a low_threshold of 50 and high_threshold of 150 for Canny edge detection.
#
# For region selection, I defined vertices =
# np.array([[(0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)
#
# I chose parameters for my Hough space grid to be a rho of 2 pixels and theta of 1 degree (pi/180 radians).
# I chose a threshold of 15, meaning at least 15 points in image space need to be associated with each line
# segment. I imposed a min_line_length of 40 pixels, and max_line_gap of 20 pixels.