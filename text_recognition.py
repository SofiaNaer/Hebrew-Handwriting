import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import numpy as np
from autocorrect import Speller

class text_recognition:
    def __init__(self, img):
        self.img = img
        self.model = keras.models.load_model("../neural-network/saved_model")

    # apply all image processing on 'img'
    def preprocess (self):

    # returns image of one sentence
    def split_lines(self, img ):

    # returns squares of letters or spaces
    def split_squares (self, line):

    # apply model and returns string
    def predict_Letter(self):










# 0. main

# 1. load model

# 2. load image
# 2.1   pre-process the image - BW, opening,...
# 2.2   split image to lines (ask GPT)

# 3. to each line: split into squares of letters
# 3.1   add spaces

# 4. to each square: apply model

# 5. apply spell checking





