import cv2
import numpy as np
import os

def check_image(img, text):
    cv2.namedWindow(text, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(text, img)
    cv2.resizeWindow(text, 1000, 220)
    cv2.waitKey(0)

def split_squares( line_path):

    output_folder = "Squares"
    space_threshold = 100  # Set the threshold distance to differentiate between letters and spaces (pixels)
    os.makedirs(output_folder, exist_ok=True)
    src_image = cv2.imread(line_path)

    src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    # image = image.astype(np.uint8)

    # Identify text regions using contour detection and sort the contours from left to right
    contours, _ = cv2.findContours(src_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0], reverse=True)

    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])

        # Check if contour is too small or too wide, it might be noise
        if w < 20 or h < 20 or w // h > 4 or w / h < 0.1:
            continue

        letter_image = src_image[y:y + h, x:x + w]
        x1 = x
        x2 = x + w
        y1 = y
        y2 = y + h

        cv2.rectangle(src_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Check the distance between the current contour and the next contour
        if i < len(contours) - 1:
            next_x, _, _, _ = cv2.boundingRect(contours[i + 1])
            distance = (x + w) - next_x
            print(distance)

            if distance > space_threshold:  # create a blank space image
                space_image = np.ones_like(letter_image) * 255
                output_path = os.path.join(output_folder, f"letter_{i}_space.jpg")
                cv2.imwrite(output_path, space_image)
                continue

        # Save the letter
        output_path = os.path.join(output_folder, f"letter_{i}.jpg")
        cv2.imwrite(output_path, letter_image)


    image = cv2.imread('handwriting_image.jpg', 0)  # Convert to grayscale
    check_image(src_image, "with boxes")




# sum all distances
# sum amount of distances
# calculate sum all distances/sum amount of distances = avarage_with_spaces
# mul average_with_spaces by 1.3 = average_space_width
# calculate words_amount = amount of distances/4
# average_space_width * words_amount = all_spaces
# all distances - all_spaces = total_width_without_spaces

# check two things:
# 1. space - 85% background
# 2. otherwise - 2 or more letters:
#   2.1 erosion
#   2.2 findContours
#   2.3 return boxes

def find_threshold (img_path):
    space_threshold = 110
    src_image = cv2.imread(img_path)
    src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(src_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0], reverse=True)
    squares_amount = 0
    total_distance = 0
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        if w < 20 or h < 20 or w // h > 4 or w / h < 0.1:
            continue
        if i < len(contours) - 1:
            next_x, _, _, _ = cv2.boundingRect(contours[i + 1])
            distance = (x + w) - next_x
            total_distance += distance
            squares_amount+=1
    print(total_distance)
    print(squares_amount)

    threshold = (total_distance/squares_amount) * 1.3
    print(threshold)
    return threshold


find_threshold("lines/0b10.jpg")

