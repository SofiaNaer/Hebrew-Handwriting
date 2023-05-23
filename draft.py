import cv2
import numpy as np

# Load the image
image = cv2.imread('lines/0b10.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate the horizontal projection profile
projection = np.sum(gray, axis=1)
print(projection, type(projection), projection.shape)

# Determine the threshold to identify the line
threshold = np.max(projection) * 0.2  # Adjust the threshold value as needed

# Detect the line region based on the projection profile
line_indices = np.where(projection > threshold)[0]
print(line_indices)
line_top = np.min(line_indices)
line_bottom = np.max(line_indices)

# Remove the line region from the image
result = np.copy(image)
result[line_top:line_bottom, :] = 0  # Set line region to white (or any desired background color)

# Display the result
cv2.imshow('Handwritten Text with Line', gray)
cv2.imshow('Handwritten Text without Line', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
