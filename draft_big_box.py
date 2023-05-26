import cv2
import numpy as np


def check_image(input_img, txt):
    cv2.namedWindow(txt, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(txt, input_img)
    cv2.resizeWindow(txt, 800, 400)
    cv2.waitKey(0)


def process_big_box(image):
    # # Convert image to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    # # Thresholding
    # _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    check_image(image, "before")

    # Calculate the percentage of black pixels
    black_pixels = np.sum(image == 0)
    total_pixels = image.shape[0] * image.shape[1]
    black_percentage = (black_pixels / total_pixels) * 100

    # if black_percentage > 95:
    #     # Mostly background (space between words)
    #     return image, 'space'

    # Erosion
    kernel = np.ones((5, 2), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=3)
    kernel2 = np.ones((2, 6), np.uint8)
    eroded = cv2.erode(dilated, kernel2, iterations=2)
    check_image(eroded, "dilated")

    # Find contours
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out noise and draw contours
    min_contour_area = 50
    letter_boxes = []
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(eroded, (x, y), (x + w, y + h), (255, 255, 255), 2)
            letter_boxes.append((x, y, w, h))

    check_image(eroded, 'Result')

    return letter_boxes
