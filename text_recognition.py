import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import numpy as np
from autocorrect import Speller


class text_recognition:
    def __init__(self, img_path):
        self.img = cv2.imread(img_path)
        self.model = keras.models.load_model("./neural-network/saved_model")
        self.preprocess()
        self.split_lines(self.img)

    # apply all image processing on 'img'
    def preprocess(self):
        # convert to to grayscale
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # apply bilateral filter to remove noise while preserving edges
        blur = cv2.bilateralFilter(gray, 9, 75, 75)

        # threshold image to create a binary image
        _, thresh = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # remove noise by opening (erosion followed by dilation)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # apply dilation
        dilation = cv2.dilate(opening, kernel, iterations=2)

        self.img = dilation
        self.check_image(dilation, "afterwards")

    @staticmethod
    def check_image(img, text):
        cv2.namedWindow(text, cv2.WINDOW_KEEPRATIO)
        cv2.imshow("sentence after pre-processing", img)
        cv2.resizeWindow('sentence after pre-processing', 1280, 720)
        cv2.waitKey(0)

    # returns image of one sentence
    def split_lines(self, img):
        x, w1 = img.shape
        if not os.path.exists("lines"):
            os.makedirs("lines")
        line_images = []
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            line_img = img[y:y + h, x:x + w]
            line_images.append(line_img)
        for i, line_img in enumerate(line_images):
            _, w = line_img.shape
            if w > w1 * 0.5:
                cv2.imwrite(f'lines\line_{i}.jpg', line_img)

            # new_image = f'lines\line_{i}.jpg'
            # width1, height1 = new_image.size
            # if width1 < width * 0.5:
            #     os.remove(f'lines\line_{i}.jpg')

    # returns squares of letters or spaces
    def split_squares(self, line):
        output_folder = "Squares"
        space_threshold = 15  # Set the threshold distance to differentiate between letters and spaces (pixels)
        os.makedirs(output_folder, exist_ok=True)

        # Identify text regions using contour detection and sort the contours from left to right
        contours, _ = cv2.findContours(line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

        for i, contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            letter_image = line[y:y + h, x:x + w]

            # Check the distance between the current contour and the next contour
            if i < len(contours) - 1:
                next_x, _, _, _ = cv2.boundingRect(contours[i + 1])
                distance = next_x - (x + w)

                if distance > space_threshold:  # create a blank space image
                    space_image = np.ones_like(letter_image) * 255
                    output_path = os.path.join(output_folder, f"space_{i}.jpg")
                    cv2.imwrite(output_path, space_image)
                    continue

            # Save the letter
            output_path = os.path.join(output_folder, f"letter_{x}.jpg")
            cv2.imwrite(output_path, letter_image)

    # apply model and returns string
    def predict_letter(self, letter_path):

        alphabet_dict = {0: 'א', 1: 'ב', 2: 'ג', 3: 'ד', 4: 'ה', 5: 'ו', 6: 'ז', 7: 'ח', 8: 'ט',
                         9: 'י', 10: 'כ', 11: 'ך', 12: 'ל', 13: 'מ', 14: 'ם', 15: 'נ', 16: 'ן', 17: 'ס', 18: 'ע',
                         19: 'פ', 20: 'ף', 21: 'צ', 22: 'ץ', 23: 'ק', 24: 'ר', 25: 'ש', 26: 'ת'}
        img_height = 28
        img_width = 28

        # process the letter box: resize and convert to array
        image = tf.keras.preprocessing.image.load_img(letter_path, target_size=(img_height, img_width))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        # input_arr = np.array([input_arr])

        # perform OCR
        predictions = self.model.predict(input_arr)
        class_index = np.argmax(predictions)
        prediction_value = np.max(predictions)

        return [alphabet_dict[class_index], prediction_value]


# 0. main

# 1. load model

# 2. load image
# 2.1   pre-process the image - BW, opening,...
# 2.2   split image to lines (ask GPT)

# 3. to each line: split into squares of letters
# 3.1   add spaces

# 4. to each square: apply model

# 5. apply spell checking

text_rec = text_recognition("sentences/sentence3.jpg")
