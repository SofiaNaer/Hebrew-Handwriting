import cv2

image = cv2.imread(
    'neural-network\processing_templates\elka-template-01.jpg', cv2.IMREAD_GRAYSCALE)
_, binarized = cv2.threshold(image, 127, 255, cv2.THRESH_OTSU)

height, width = binarized.shape

square_width = width // 9
square_height = height // 13

# Loop through each square and save it as a separate image
for row in range(13):
    for col in range(9):

        # Calculate the coordinates of the square
        left = col * square_width
        top = row * square_height
        right = left + square_width
        bottom = top + square_height

        square = binarized[top:bottom, left:right]

        cv2.imwrite(
            f"neural-network/processing_templates/outputs_from_templates/output_square_{row}_{col}.jpg", square)
