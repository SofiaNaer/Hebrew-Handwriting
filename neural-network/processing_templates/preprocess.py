import os
import cv2
from split_image import  Split

class preprocess:
    def __init__(self):

        self.input_folder = "filled_in_templates"
        self.result_folder_name = 'after_preprocessing'
        self.new_folder_path = ""
        self.prepare_pict()



    def prepare_pict (self):
        for folder_path, subfolders, filenames in os.walk(self.input_folder):

            count = 0
            for filename in filenames:
                self.new_folder_path = os.path.join(folder_path, self.result_folder_name)
                if not os.path.exists(self.new_folder_path):
                    os.makedirs(self.new_folder_path)

                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):

                    image = cv2.imread(os.path.join(folder_path, filename),
                                       cv2.IMREAD_GRAYSCALE)
                    _, binarized = cv2.threshold(
                        image, 127, 255, cv2.THRESH_BINARY_INV)


                    # Define a structuring element for morphology operations
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    # Apply morphology operations to remove the grid
                    morph = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel)

                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    binarized = cv2.dilate(binarized, kernel, iterations=3)

                    height, width = binarized.shape
                    binarized = binarized[int(
                        height - height * 0.98):int(0.98 * height), int(width - width*0.99):int(width * 0.94)]

                    # cv2.namedWindow('preview', cv2.WINDOW_KEEPRATIO)
                    # cv2.resizeWindow('preview', 500, 1000)
                    # cv2.imshow('preview', binarized)
                    # cv2.waitKey(0)

                    output_filename = str(count) + ".jpg"
                    count+=1
                    output_path = os.path.join(
                        self.new_folder_path, output_filename)
                    cv2.imwrite(output_path, binarized)


            # print(self.new_folder_path)

            # print(new_folder_path)
        split = Split(self.new_folder_path)







guy = preprocess();