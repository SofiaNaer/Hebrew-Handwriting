import os
import cv2

# Define the input and output folders
input_folder = "neural-network\processing_templates\\filled_in_templates"
output_folder = "neural-network\processing_templates\\results"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all subfolders in the input folder
for root, dirs, files in os.walk(input_folder):
    for filename in files:

        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):

            image = cv2.imread(os.path.join(root, filename),
                               cv2.IMREAD_GRAYSCALE)
            _, binarized = cv2.threshold(
                image, 127, 255, cv2.THRESH_BINARY_INV)

            # cv2.imshow("im", binarized)
            # cv2.waitKey(0)

            height, width = binarized.shape
            square_width = width // 9
            square_height = height // 13

            binarized = binarized[0:height, 0:int(width * 0.95)]

            # cv2.imshow("im", binarized)
            # cv2.waitKey(0)

            for row in range(13):
                for col in range(9):

                    # Calculate the coordinates of the square
                    left = col * square_width
                    top = row * square_height
                    right = left + square_width
                    bottom = top + square_height

                    # Crop the binarized image to the square
                    square = binarized[top:bottom, left:right]

                    # Save the square as a JPEG file
                    output_filename = f"{os.path.splitext(filename)[0]}_square_{row}_{col}.jpg"
                    output_path = os.path.join(output_folder, output_filename)
                    print(output_path)
                    cv2.imwrite(output_path, square)
