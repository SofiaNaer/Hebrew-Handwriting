import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import numpy as np


alphabet_dict = {0: 'א', 1: 'ב', 2: 'ג', 3: 'ד', 4: 'ה', 5: 'ו', 6: 'ז', 7: 'ח', 8: 'ט',
                    9: 'י', 10: 'כ', 11: 'ן', 12: 'ל', 13: 'מ', 14: 'ם', 15: 'נ', 16: 'ן', 17: 'ס', 18: 'ע', 19: 'פ', 20: 'ף', 21: 'צ', 22: 'ץ', 23: 'ק', 24: 'ר', 25: 'ש', 26: 'ת'}
img_height = 28
img_width = 28

# load model
model = keras.models.load_model("neural-network/saved_model")


def predict_letter(img_path, index):

    # pre-processing the image:
    image = tf.keras.preprocessing.image.load_img(
        img_path, color_mode="grayscale", target_size=(img_height, img_width))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])

    # predict
    predictions = model.predict(input_arr)
    # alternative way: (why is it faster?)
    # predictions = model(input_arr)

    class_index = np.argmax(predictions)
    prediction_value = np.max(predictions)

    # print predictions results
    print(f"prediction ({index}): " + str(alphabet_dict[class_index]) +
          "     confidence level: " + str(prediction_value) + "\n")

    return [alphabet_dict[class_index], prediction_value]


# Load image and convert to grayscale
img = cv2.imread('prototype/sentences/ex_sentence.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Apply bilateral filter to remove noise while preserving edges
blur = cv2.bilateralFilter(gray, 9, 75, 75)

# Threshold image to create a binary image
_, thresh = cv2.threshold(
    blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


# Remove noise by opening (erosion followed by dilation)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# connect the letters which contain two parts:
# Apply dilation to merge contours
dilation = cv2.dilate(opening, kernel, iterations=2)

cv2.namedWindow('sentence after pre-processing', cv2.WINDOW_KEEPRATIO)
cv2.imshow("sentence after pre-processing", dilation)
cv2.resizeWindow('sentence after pre-processing', 1000, 500)
cv2.waitKey(0)

# Find contours of the individual letters
contours, hierarchy = cv2.findContours(
    dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours right to left to identify spaces between words
contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[
                  0] + cv2.boundingRect(ctr)[2], reverse=True)


# Create a directory to save the cropped images
if not os.path.exists('prototype/letters'):
    os.makedirs('prototype/letters')

# merged_contours = []
# min_x_overlap = 5

# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     if len(merged_contours) > 0:
#         prev_x, prev_y, prev_w, prev_h = merged_contours[-1]
#         if x - prev_x <= min_x_overlap:
#             merged_contours[-1] = (prev_x, min(prev_y, y),
#                                    x + w - prev_x, max(prev_h, h))
#             continue
#     merged_contours.append((x, y, w, h))


result = ''
total_value = 0

# print(contours)

# Iterate through the contours and crop each letter
for i, contour in enumerate(contours):
    # Get bounding box of contour
    (x, y, w, h) = cv2.boundingRect(contour)
    h = h * 2

    # Check if contour is too small or too wide, it might be noise
    if w < 7 or h < 7 or w//h > 4 or w/h < 0.1:
        continue

    print("\nwidth: " + str(w) + " height: " + str(h))
    print("aspect ratio: " + str(w/h))

    # box coordinates
    x1 = x
    x2 = x + w
    y1 = y - h//4
    y2 = y + h//2

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Crop the letter and save as a JPEG
    letter = dilation[y1:y2, x1:x2]
    cv2.imwrite(f'prototype/letters/letter{i}.jpg', letter)

    # use current model to predict the character
    prediction, value = predict_letter(f"prototype/letters/letter{i}.jpg", i)
    result += prediction
    total_value += value

    # Add a space after each letter
    if i < len(contours) - 1:
        next_x = cv2.boundingRect(contours[i+1])[0]
        space_width = x - (next_x + cv2.boundingRect(contours[i+1])[2])
        print("space width: " + str(space_width))

        if space_width > 1.5 * w:
            space = 255 * np.ones((h, space_width), np.uint8)
            result += ' '
            cv2.imwrite(f'prototype/letters/letter{i}_space.jpg', space)

cv2.namedWindow('image with boxes', cv2.WINDOW_KEEPRATIO)
cv2.imshow("image with boxes", img)
cv2.resizeWindow('image with boxes', 1280, 720)
cv2.waitKey(0)

result = result[::-1]
print("result: " + result)
print("average confidence level: " + str(round(total_value/i, 5)))
cv2.destroyAllWindows()
