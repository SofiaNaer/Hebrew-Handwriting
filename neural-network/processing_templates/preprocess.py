import os
import cv2



input_folder = "filled_in_templates"
result_folder_name = 'after_preprocessing'

for folder_path, subfolders, filenames in os.walk(input_folder):
    new_folder_path = os.path.join(folder_path, result_folder_name)
    os.makedirs(new_folder_path, exist_ok=True)
    count = 0
    for filename in filenames:

        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):

            image = cv2.imread(os.path.join(folder_path, filename),
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



            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            binarized = cv2.dilate(binarized, kernel, iterations=3)

            height, width = binarized.shape
            binarized = binarized[int(
                height - height * 0.98):int(0.98 * height), int(width - width*0.99):int(width * 0.94)]

            cv2.namedWindow('preview', cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow('preview', 500, 1000)
            cv2.imshow('preview', binarized)
            cv2.waitKey(0)

            output_filename = str(count) + ".jpg"
            count+=1
            output_path = os.path.join(
                new_folder_path, output_filename)
            print(output_path)
            cv2.imwrite(output_path, binarized)
