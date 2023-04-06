import numpy as np
import cv2
import os

# Load the document image
image = cv2.imread('prototype\sentences\ex_sentence.jpg',
                   cv2.IMREAD_GRAYSCALE)

# Perform thresholding to create a binary image
_, binary_image = cv2.threshold(
    image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# Remove noise by opening (erosion followed by dilation)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

# Close small holes by closing (dilation followed by erosion)
binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

# Apply dilation to better segment the characters
binary_image = cv2.dilate(binary_image, kernel, iterations=1)

# Set up the blob detector parameters
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 80
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False

# Create the blob detector
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs in the binary image
keypoints = detector.detect(binary_image)

# Draw circles around the blobs to visualize the detected characters
image_with_blobs = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

if not os.path.exists('./prototype/letters/'):
    os.makedirs('./prototype/letters/')

for i, keypoint in enumerate(keypoints):
    x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
    width = int(keypoint.size * 1.1)  # Increase the width of the rectangle
    height = int(keypoint.size * 3)  # Increase the height of the rectangle

    # Calculate the coordinates of the top-left and bottom-right corners of the rectangle
    x1, y1 = x - width // 2, y - height // 2
    x2, y2 = x + width // 2, y + height // 2

    # Skip blobs that are too small or too large to be characters
    if width < 10 or height < 10:
        continue

    cv2.rectangle(image_with_blobs, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Extract character segment from morphed image
    character_segment = binary_image[y1:y2, x1:x2]
    character_segment_path = os.path.join(
        './prototype/letters', f'character_{i+1}.png')
    # Save character segment as an image
    # cv2.imwrite(character_segment_path, character_segment)

# # Display the original image and the image with blobs
# cv2.namedWindow('Original Image', cv2.WINDOW_KEEPRATIO)
# cv2.resizeWindow('Original Image', 1280, 720)
# cv2.imshow('Original Image', binary_image)


cv2.namedWindow('Image with Blobs', cv2.WINDOW_KEEPRATIO)
cv2.imshow('Image with Blobs', image_with_blobs)
cv2.resizeWindow('Image with Blobs', 1280, 720)

cv2.waitKey(0)
cv2.destroyAllWindows()
