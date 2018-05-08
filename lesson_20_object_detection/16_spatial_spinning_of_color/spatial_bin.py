import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in an image
# You can also read cutout2, 3, 4 etc. to see other examples
# image = mpimg.imread('cutout1.jpg')

image_list = ['./cutouts/cutout1.jpg', './cutouts/cutout2.jpg', './cutouts/cutout3.jpg',
              './cutouts/cutout4.jpg', './cutouts/cutout5.jpg', './cutouts/cutout6.jpg']

image_list = image_list + ['others/31-car.png', 'others/53-car.png', 'others/2-nocar.png',
                           'others/8-nocar.png', 'others/3-nocar.png', 'others/25-car.png']

# Define a function to compute color histogram features
# Pass the color_space flag as 3-letter all caps string
# like 'HSV' or 'LUV' etc.
# KEEP IN MIND IF YOU DECIDE TO USE THIS FUNCTION LATER
# IN YOUR PROJECT THAT IF YOU READ THE IMAGE WITH
# cv2.imread() INSTEAD YOU START WITH BGR COLOR!
def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image tonew color space (if specified)
    if color_space == 'HSV':
        img_copy = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif color_space == 'LUV':
        img_copy = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    elif color_space == 'HLS':
        img_copy = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif color_space == 'LAB':
        img_copy = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    else:
        img_copy = np.copy(img)
    # Use cv2.resize().ravel() to create the feature vector
    dst = cv2.resize(img_copy, size)

    features = dst.ravel()

    # Return the feature vector
    return features

color_spaces = ['RGB', 'HSV', 'LUV', 'HLS', 'LAB']

for image_name in image_list:
    for color_space in color_spaces:
        image = mpimg.imread(image_name)

        feature_vec = bin_spatial(image, color_space=color_space, size=(32, 32))

        # Plot features
        plt.plot(feature_vec)
        plt.title('Spatially Binned Features: {} - {}'.format(image_name, color_space))
        plt.show()