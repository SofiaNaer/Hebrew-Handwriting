import numpy as np
import cv2
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

# load the image
image = cv2.imread("./prototype/sentences/ex_sentence.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, image = cv2.threshold(
    image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# apply SLIC segmentation to the image
segments = slic(image, n_segments=100, compactness=10,
                sigma=1, channel_axis=None)

# loop over the segments
for i in np.unique(segments):
    # create a mask for the segment
    mask = np.zeros(image.shape[:2], dtype="uint8")
    mask[segments == i] = 255

    # find the contours in the mask
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # if there is only one contour, draw it on the image
    if len(contours) == 1:
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    # otherwise, split the segment into sub-segments and draw them on the image
    else:
        sub_segments = slic(image[segments == i], n_segments=len(
            contours), compactness=10, sigma=1)
        for j in np.unique(sub_segments):
            sub_mask = np.zeros(image.shape[:2], dtype="uint8")
            sub_mask[sub_segments == j] = 255
            sub_contours, sub_hierarchy = cv2.findContours(
                sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, sub_contours, -1, (0, 255, 0), 2)

# display the segmented image
cv2.namedWindow('Segmented Image', cv2.WINDOW_KEEPRATIO)
cv2.imshow("Segmented Image", image)
cv2.resizeWindow('Segmented Image', 1000, 500)
cv2.waitKey(0)
