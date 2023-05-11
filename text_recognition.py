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
       # self.model = keras.models.load_model("../neural-network/saved_model")
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
        pass

    # apply model and returns string
    def predict_letter(self):
        pass

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