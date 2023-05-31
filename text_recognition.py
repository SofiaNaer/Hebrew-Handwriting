import os
import shutil
from fileinput import filename
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import numpy as np
from autocorrect import Speller
from PIL import Image
from PIL import ImageEnhance
from Split_letters import Split_letters
import random


class text_recognition:
    def __init__(self, img_path):
        self.result = ""
        self.img = cv2.imread(img_path)
        print("original image:", type(self.img))
        self.model = keras.models.load_model("./neural-network/saved_model")
        self.black_and_white= self.preprocess1()
        self.split_lines(self.black_and_white)
        self.convert_lines_to_letters("./lines")
        print('result:\n', self.result)
        #print(self.spell_check())

        # apply all image processing on 'img'
    def preprocess1(self):
        black_white_img = self.img.copy()
        # convert to to grayscale
        gray = cv2.cvtColor(black_white_img, cv2.COLOR_BGR2GRAY)
        # apply bilateral filter to remove noise while preserving edges
        blur = cv2.bilateralFilter(gray, 9, 75, 75)
        # threshold image to create a binary image
        _, thresh = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # remove noise by opening (erosion followed by dilation)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # apply dilation
        # dilation = cv2.dilate(thresh, kernel, iterations=1)

        black_white_img = thresh
        self.check_image(black_white_img, "afterwards")
        return  black_white_img

    def preprocess(self, img):
        # convert to to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # apply bilateral filter to remove noise while preserving edges
        blur = cv2.bilateralFilter(gray, 9, 75, 75)
        # threshold image to create a binary image
        _, thresh = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # remove noise by opening (erosion followed by dilation)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # apply dilation
        dilation = cv2.dilate(thresh, kernel, iterations=2)
        self.check_image(dilation, "afterwards")
        return dilation

    # returns image of one sentence
    def split_lines(self, img):
        print("split_lines image",type(img))
        h1, w1 = img.shape
        if not os.path.exists("lines"):
            os.makedirs("lines")
        line_images = []
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        j = 1
        for i in range(len(contours) - 1, -1, -1):
            contour = contours[i]
            x, y, w, h = cv2.boundingRect(contour)
            line_img = self.img[y:y + h, x:x + w]
            line_images.append(line_img)

        for i in range(len(line_images)):
            h, w, _ = line_images[i].shape

            if w > w1 * 0.5 and h > h1 * 0.05:
                #bin_i = bin(i)
                # without_line = self.remove_bottom_line(line_images[i])

                cv2.imwrite(f'lines\\{j}.jpg', line_images[i] )
                self.brighten_image(f'lines\\{j}.jpg', 2)
                j += 1


        # returns squares of letters or spaces

    def brighten_image(self, image_path, brightness_factor):
        # Open the image file
        image = Image.open(image_path)
        # Create an enhancer object and apply brightness enhancement
        enhancer = ImageEnhance.Brightness(image)
        brightened_image = enhancer.enhance(brightness_factor)
        brightened_image_array = np.array(brightened_image)
        # output_path = "brightened_image.jpg"
        print("brighten image", type(brightened_image_array))
        bw_sentence = self.preprocess(brightened_image_array)
        cv2.imwrite(image_path, bw_sentence)




    # create letter squares in a folder named "Squares"
    def convert_lines_to_letters(self, lines_folder):
        for filename in os.listdir(lines_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                filepath = os.path.join(lines_folder, filename)
                split_l = Split_letters(filepath)
                print(filepath)
                self.result += self.send_to_OCR("Squares") + '\n'
                #self.delete_folder("Squares")
                self.move_folder("Squares")

    def send_to_OCR(self, letters_folder):
        result = ""
        sorted_images = self.sort_numerically(letters_folder)
        print(sorted_images)

        for filename in sorted_images:
            if "space" in filename:
                result += " "
            else:
                filepath = os.path.join(letters_folder, filename)
                character, value = self.predict_letter(filepath)
                result += character

        print(result)
        return result

    # apply model and returns string
    def predict_letter(self, letter_path):

        alphabet_dict = {0: 'א', 1: 'ב', 2: 'ג', 3: 'ד', 4: 'ה', 5: 'ו', 6: 'ז', 7: 'ח', 8: 'ט',
                         9: 'י', 10: 'כ', 11: 'ך', 12: 'ל', 13: 'מ', 14: 'ם', 15: 'נ', 16: 'ן', 17: 'ס', 18: 'ע',
                         19: 'פ', 20: 'ף', 21: 'צ', 22: 'ץ', 23: 'ק', 24: 'ר', 25: 'ש', 26: 'ת'}
        img_height = 28
        img_width = 28

        # process the letter box: resize and convert to array
        image = tf.keras.preprocessing.image.load_img(letter_path, grayscale='true',
                                                      target_size=(img_height, img_width))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])

        # perform OCR
        predictions = self.model.predict(input_arr)
        class_index = np.argmax(predictions)
        prediction_value = np.max(predictions)

        return [alphabet_dict[class_index], prediction_value]

    def spell_check(self):
        Spell = Speller('he')
        return Spell(self.result)

    def remove_bottom_line(self, image):

        # image = cv2.imread(img)
        # self.check_image(image,"xhexk")
        # Convert the image to grayscale
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate the horizontal projection profile
        projection = np.sum(image, axis=1)
        print(projection, type(projection), projection.shape)

        # Determine the threshold to identify the line
        threshold = np.max(projection) * 0.2  # Adjust the threshold value as needed

        # Detect the line region based on the projection profile
        line_indices = np.where(projection > threshold)[0]
        line_top = np.min(line_indices)
        line_bottom = np.max(line_indices)

        # Remove the line region from the image
        result = np.copy(image)
        result[line_top:line_bottom, :] = 0  # Set line region to white (or any desired background color)
        return result

        # Display the result
        # cv2.imshow('Handwritten Text with Line', gray)
        # cv2.imshow('Handwritten Text without Line', result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    @staticmethod
    def check_image(img, text):
        cv2.namedWindow(text, cv2.WINDOW_KEEPRATIO)
        cv2.imshow(text, img)
        cv2.resizeWindow(text, 1280, 200)
        cv2.waitKey(0)

    def delete_folder(self, path):
        try:
            # remove the folder and all its contents
            shutil.rmtree(path)
            print("Folder deleted successfully")
        except OSError as error:
            print(f"Error: {path} : {error.strerror}")

    def move_folder(self, source):
        random_number = random.randint(1, 1000)
        new_name = f"{random_number}"

        # Rename the folder
        os.rename(source, new_name)



text_rec = text_recognition("sentences/sentence5.jpg")
