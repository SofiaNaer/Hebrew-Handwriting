import cv2

image = cv2.imread('prototype/words/inline.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Apply thresholding to binarize the image
thresh = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Find contours of the individual letters
contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Define the range of aspect ratios for valid letters
min_aspect_ratio = 0.2
max_aspect_ratio = 1.5
min_area = 10
max_area = 5000

cv2.imshow('word: ', thresh)
cv2.waitKey(0)

for i, contour in enumerate(contours):
    # Get the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(contour)

    # Calculate the aspect ratio and area of the bounding box
    aspect_ratio = float(w) / h
    area = w * h

    # Check if the aspect ratio and area are within the valid range
    if min_aspect_ratio < aspect_ratio < max_aspect_ratio and min_area < area < max_area:
        # Check if the contour is approximately rectangular
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)

        if len(approx) == 4:
            letter = image[y:y+h, x:x+w]
            cv2.imwrite(f'prototype/letter_{i}.jpg', letter)

            cv2.imshow('Letter {}'.format(i+1), letter)
            cv2.waitKey(0)

# Wait for a key press and then close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
