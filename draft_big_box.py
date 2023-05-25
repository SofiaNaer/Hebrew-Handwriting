import cv2
import numpy as np


def process_big_box(image):
    # # Convert image to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    # # Thresholding
    # _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Calculate the percentage of black pixels
    black_pixels = np.sum(image == 0)
    total_pixels = image.shape[0] * image.shape[1]
    black_percentage = (black_pixels / total_pixels) * 100

    if black_percentage > 85:
        # Mostly background (space between words)
        return image, 'space'

    # Erosion
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(image, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out noise
    min_contour_area = 50  # Adjust this threshold as per your requirements
    letter_boxes = []
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(eroded, (x, y), (x + w, y + h), (0, 255, 0), 2)
            letter_boxes.append((x, y, w, h))

    cv2.imshow('Result', eroded)
    cv2.waitKey(0)

    return image, letter_boxes
