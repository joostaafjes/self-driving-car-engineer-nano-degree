import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.ion()
image = mpimg.imread('bbox-example-image.jpg')
# image = mpimg.imread('temp-matching-example-2.jpg')
templist = ['./cutouts/cutout1.jpg', './cutouts/cutout2.jpg', './cutouts/cutout3.jpg',
            './cutouts/cutout4.jpg', './cutouts/cutout5.jpg', './cutouts/cutout6.jpg']

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Define a function that takes an image and a list of templates as inputs
# then searches the image and returns the a list of bounding boxes
# for matched templates
def find_matches(img, template_list, method):
    # Make a copy of the image to draw on
    img_copy = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Define an empty list to take bbox coords
    bbox_list = []
    # Iterate through template list
    method = eval(method)
    for template in template_list:
        template_img = cv2.imread(template, 0)
        w, h = template_img.shape[::-1]

        result = cv2.matchTemplate(img_copy, template_img, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        bbox_list.append([top_left, bottom_right])

    return bbox_list

for method in methods:
    bboxes = find_matches(image, templist, method)
    result = draw_boxes(image, bboxes)
    plt.imshow(result)
    plt.title("method : " + method)
    plt.show()
    # input('press key to continue...')