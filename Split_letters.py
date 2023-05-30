import cv2
import numpy as np
import os

class Split_letters:
    def __init__(self, line_path):

        self.split_squares(line_path)



    def process_big_box(self, image, index):
        # # Convert image to grayscale
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #
        # # Thresholding
        # _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # check_image(image, "before")

        # Erosion
        min_contour_area = 50
        kernel = np.ones((5, 2), np.uint8)
        dilated = cv2.dilate(image, kernel, iterations=2)
        kernel2 = np.ones((2, 6), np.uint8)
        eroded = cv2.erode(dilated, kernel2, iterations=2)
        self.check_image(eroded, "dilated")

        # Find contours
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out noise and draw contours
        for contour in contours:
            if cv2.contourArea(contour) > min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)
                sub_letter = eroded[y:y + h, x:x + w]
                cv2.imwrite(f'./Squares/letter_{index}.jpg', sub_letter)
                index += 1

        self.check_image(image, 'Result')

        return index


    def check_image(self, img, text):
        cv2.namedWindow(text, cv2.WINDOW_KEEPRATIO)
        cv2.imshow(text, img)
        cv2.resizeWindow(text, 800, 400)
        cv2.waitKey(0)


    def split_squares(self, line_path):
        output_folder = "Squares"
          # Set the threshold distance to differentiate between letters and spaces (pixels)
        os.makedirs(output_folder, exist_ok=True)
        src_image = cv2.imread(line_path)

        src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
        # image = image.astype(np.uint8)

        # Identify text regions using contour detection and sort the contours from left to right
        contours, _ = cv2.findContours(src_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0], reverse=True)
        j = 0
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            # Check if contour is too small or too wide, it might be noise
            if w < 20 or h < 20 or w // h > 4 or w / h < 0.1:
                continue
            filtered_contours.append(contours[i])

        space_threshold, avg_width = self.find_threshold(filtered_contours)
        print (avg_width, "avg_width")

        for i in range(len(filtered_contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            j += 1

            letter_image = src_image[y:y + h, x:x + w]
            x1 = x
            x2 = x + w
            y1 = y
            y2 = y + h

            # cv2.rectangle(src_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # Check the distance between the current contour and the next contour
            if i < len(filtered_contours) - 1:

                next_x, _, next_w, _ = cv2.boundingRect(filtered_contours[i+1])
                print(x, "current x")
                print(w, "current w")
                print(next_x, "next x")
                print(next_w, "next w")


                distance = x - next_x - next_w
               # letter_width = self.threshold * 1.2

                print(distance)


                if distance > space_threshold - 5:
                    #flag = True
                    print(w, "width")
                    print("it's a space")
                    space_image = np.ones_like(letter_image) * 255
                    output_path = os.path.join(output_folder, f"letter_{j}_space.jpg")
                    cv2.imwrite(output_path, space_image)

                    continue

            # Save the letter
                if not flag:
                    output_path = os.path.join(output_folder, f"letter_{j}.jpg")
                    print("it's normal")
                    cv2.imwrite(output_path, letter_image)

        self.check_image(src_image, "with boxes")

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

    def find_threshold(self, contours):

        squares_amount = 0
        total_distance = 0

        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            if w < 20 or h < 20 or w // h > 4 or w / h < 0.1:
                continue
            if i < len(contours) - 1:
                next_x, _, next_w, _ = cv2.boundingRect(contours[i + 1])
                distance = x - next_x

                # distance = w + (next_x

                total_distance += distance
                squares_amount += 1

                total_w += w



        threshold = (total_distance  / (squares_amount + 1))
        print(int(threshold), "it's threshold")
        #avarage_letter_width = (total_distance - threshold * (squares_amount/4))/squares_amount
        return int(threshold), total_w//squares_amount


