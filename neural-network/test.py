import tensorflow as tf
from tensorflow import keras
import numpy as np

model = keras.models.load_model("neural-network/saved_model")

alphabet_dict = {0: 'א', 1: 'ב', 2: 'ג', 3: 'ד', 4: 'ה', 5: 'ו', 6: 'ז', 7: 'ח', 8: 'ט',
                 9: 'י', 10: 'כ', 11: 'ן', 12: 'ל', 13: 'מ', 14: 'ם', 15: 'נ', 16: 'ן', 17: 'ס', 18: 'ע', 19: 'פ', 20: 'ף', 21: 'צ', 22: 'ץ', 23: 'ק', 24: 'ר', 25: 'ש', 26: 'ת'}

img_height = 28
img_width = 28

print("Predictions:")
image = tf.keras.preprocessing.image.load_img(
    "dataset/6/0069_dilated.jpg", color_mode="grayscale", target_size=(img_height, img_width))

input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = model.predict(input_arr)

print(predictions)
print("the letter index is: " + str(np.argmax(predictions)))
print("the predited result: " + alphabet_dict.get(np.argmax(predictions)))
