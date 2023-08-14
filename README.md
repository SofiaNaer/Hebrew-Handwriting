# Hebrew Handwriting Recognition and Spell Checking

This project focuses on Hebrew handwriting recognition and spell checking using an autocorrect library. It provides a Python-based solution for recognizing handwritten Hebrew text and applying spell checking using an external library. The project utilizes image processing techniques and a neural network model for character recognition.
Users have the option to improve the handwriting recognition accuracy by providing their own handwriting samples and retraining the neural network.
## Getting Started

To use this project, follow these steps:

1. Clone the repository to your local machine:


2. Install the required dependencies:


3. Place your input image containing handwritten Hebrew text in the `sentences` directory.

4. Run the `text_recognition.py` script:


## Features

- Preprocesses input images to enhance text visibility and remove noise.
- Splits text into lines and individual letters for recognition.
- Utilizes a pre-trained neural network model for Hebrew character recognition.
- Applies spell checking using an autocorrect library.

## Code Overview

The `text_recognition.py` script provides the main functionality of the project:

- `text_recognition` class: Handles image preprocessing, text splitting, character recognition, and spell checking.
- `preprocess1` method: Applies initial preprocessing to the input image.
- `split_lines` method: Segments the text into lines based on contours.
- `convert_lines_to_letters` method: Converts lines to individual letters and performs character recognition.
- `send_to_OCR` method: Sends letters to the OCR model for recognition.
- `spell_check` method: Applies spell checking to the recognized text.

## Dependencies

- OpenCV (`cv2`) for image preprocessing and contour detection.
- TensorFlow and Keras for neural network-based character recognition.
- `autocorrect` library for spell checking.
