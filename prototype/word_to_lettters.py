from PIL import Image
import numpy as np

# Load the image
img = Image.open("prototype/sentences/sentence2.jpg")

# Convert the image to grayscale
gray = img.convert("L")

# Convert the image to a numpy array
img_array = np.array(gray)

# Threshold the image to convert it to a binary image
threshold_value = 128
thresholded = img_array > threshold_value

# Find the start and end of each letter
letter_ranges = []
in_letter = False
for i in range(thresholded.shape[1]):
    column = thresholded[:, i]
    if column.any() and not in_letter:
        letter_ranges.append((i,))
        in_letter = True
    elif not column.any() and in_letter:
        letter_ranges[-1] = (letter_ranges[-1][0], i-1)
        in_letter = False
if in_letter:
    letter_ranges[-1] = (letter_ranges[-1][0], thresholded.shape[1]-1)

# Find the letter widths and spaces between them
letter_widths = []
for start, end in letter_ranges:
    width = end - start + 1
    letter_widths.append(width)

space_threshold = max(letter_widths) * 1.5
spaces = []
for i in range(len(letter_ranges)-1):
    end_of_current_letter = letter_ranges[i][1]
    start_of_next_letter = letter_ranges[i+1][0]
    space_width = start_of_next_letter - end_of_current_letter - 1
    if space_width > space_threshold:
        spaces.append(end_of_current_letter)

# Crop each letter and save it as a JPEG file
for i, (start, end) in enumerate(letter_ranges):
    letter_image = gray.crop((start, 0, end, gray.size[1]))
    letter_image.save(f"letter{i}.jpg")

# Save the spaces as JPEG files with a "-" in the file name
for i, space in enumerate(spaces):
    space_image = gray.crop((space, 0, space+1, gray.size[1]))
    space_image.save(f"space-{i}.jpg")
