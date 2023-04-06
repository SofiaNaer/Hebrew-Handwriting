import os
import cv2

# Define the input and output folders
input_folder = "neural-network\processing_templates\\filled_in_templates"
output_folder = "neural-network\processing_templates\\results"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

rows = 13
cols = 9


# Loop through all subfolders in the input folder
for root, dirs, files in os.walk(input_folder):

    max_letters = 243

    for filename in files:

        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):

            image = cv2.imread(os.path.join(root, filename),
                               cv2.IMREAD_GRAYSCALE)
            _, binarized = cv2.threshold(
                image, 127, 255, cv2.THRESH_BINARY_INV)

            cv2.namedWindow('preview', cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow('preview', 500, 1000)
            cv2.imshow('preview', binarized)
            cv2.waitKey(0)

            # Define a structuring element for morphology operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            # Apply morphology operations to remove the grid
            morph = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel)

            cv2.namedWindow('preview', cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow('preview', 500, 1000)
            cv2.imshow('preview', binarized)
            cv2.waitKey(0)

            # Remove noise by opening (erosion followed by dilation)
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            # binarized = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, kernel)

            # connect the letters which contain two parts:
            # Apply dilation to merge contours
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            binarized = cv2.dilate(binarized, kernel, iterations=3)

            height, width = binarized.shape
            square_width = width / cols
            square_height = height / rows

            binarized = binarized[int(
                height-height*0.98):int(0.98*height), 0:int(width * 0.94)]

            cv2.namedWindow('preview', cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow('preview', 500, 1000)
            cv2.imshow('preview', binarized)
            cv2.waitKey(0)

            for row in range(rows):
                if max_letters == 0:
                    break

                for col in range(cols):

                    # Calculate the coordinates of the square
                    left = round(col * square_width)
                    top = round(row * square_height)
                    right = round(left + square_width)
                    bottom = round(top + square_height)

                    # Crop the binarized image to the square
                    square = binarized[top:bottom, left:right]

                    # Save the square as a JPEG file
                    output_filename = f"{os.path.splitext(filename)[0]}_square_{row}_{col}.jpg"
                    output_path = os.path.join(
                        output_folder, output_filename)
                    print(output_path)
                    cv2.imwrite(output_path, square)

                    max_letters = max_letters - 1



