import cv2
import numpy as np
import os

class Split_letters:
    def __init__(self, line_path):

        self.split_squares(line_path)



    def process_big_box(self, image, index):
        height, width = image.shape
        mid_point = width//2

        first = image[:, mid_point:width]
        second = image[:, 0:mid_point]

        cv2.imwrite(f'./Squares/{index}.jpg', first)
        cv2.imwrite(f'./Squares/{index + 1}.jpg', second)



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
        filtered_contours = []
        j = 0
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            # Check if contour is too small or too wide, it might be noise
            if w < 20 or h < 20 or w // h > 4 or w / h < 0.1:
                continue
            filtered_contours.append(contours[i])

        space_threshold, avg_width = self.find_threshold(filtered_contours)
        print (avg_width, "avg_width")

        flag = False
        for i in range(len(filtered_contours)):
            print('\n')
            print(len(filtered_contours))
            x, y, w, h = cv2.boundingRect(filtered_contours[i])
            j += 1
            flag = False
            letter_image = src_image[y:y + h, x:x + w]
            x1 = x
            x2 = x + w
            y1 = y
            y2 = y + h

            if i == len(filtered_contours) - 1:
                if w > avg_width*1.4:
                    flag = True
                    self.process_big_box(letter_image, j)
                    j += 1
                else:
                    output_path = os.path.join(output_folder, f"{j}.jpg")
                    print("it's normal")
                    cv2.imwrite(output_path, letter_image)


            # cv2.rectangle(src_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # Check the distance between the current contour and the next contour
            if i < len(filtered_contours) - 1:
                next_x, _, next_w, _ = cv2.boundingRect(filtered_contours[i+1])
                print(x, "current x")
                print(w, "current w")
                print(next_x, "next x")
                print(next_w, "next w")


                distance = x - next_x - next_w
                print("distance", distance)

                if distance < -50:
                    continue
               # letter_width = self.threshold * 1.2

                print(distance)

                if w > avg_width*1.4:
                    flag = True
                    self.process_big_box(letter_image, j)
                    j += 1


                if distance > space_threshold - 10:
                    print(w, "width")
                    print("it's a space")
                    space_image = np.ones_like(letter_image) * 255
                    output_path = os.path.join(output_folder, f"{j}_space.jpg")
                    cv2.imwrite(output_path, space_image)



            # Save the letter
                if not flag:
                    output_path = os.path.join(output_folder, f"{j}.jpg")
                    print("it's normal")
                    cv2.imwrite(output_path, letter_image)


        self.check_image(src_image, "with boxes")



    def find_threshold(self, contours):

        squares_amount = 0
        total_distance = 0
        total_w = 0

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
                avg_letter = total_w//squares_amount

        if avg_letter < 87:
            avg_letter = 87

        threshold = (total_distance / (squares_amount +1))
        if threshold < 90:
            threshold = 90
        print(int(threshold), "it's threshold")
        print(int(avg_letter), "it's avg_letter")

        return int(threshold), int(avg_letter)


