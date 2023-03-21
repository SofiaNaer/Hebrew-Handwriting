import os
import cv2
import numpy as np

# Load image and convert to grayscale
img = cv2.imread('prototype/sentences/sentence2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply bilateral filter to remove noise while preserving edges
blur = cv2.bilateralFilter(gray, 9, 75, 75)

# Threshold image to create a binary image
_, thresh = cv2.threshold(
    blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Remove noise by opening (erosion followed by dilation)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

cv2.imshow("", opening)
cv2.waitKey(0)

# Find contours of the individual letters
contours, hierarchy = cv2.findContours(
    opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a directory to save the cropped images
if not os.path.exists('prototype/letters'):
    os.makedirs('prototype/letters')

# Iterate through the contours and crop each letter
for i, contour in enumerate(contours):
    # Get bounding box of contour
    (x, y, w, h) = cv2.boundingRect(contour)

    # Crop the letter and save as a JPEG
    letter = opening[y:y+h, x:x+w]
    cv2.imwrite(f'prototype/letters/letter{i}.jpg', letter)
