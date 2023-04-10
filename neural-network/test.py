import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

model = keras.models.load_model("neural-network/saved_model")
img_path = "processing_templates/filled_in_templates/result/cbafb93e-a1b6-435e-a1fb-019a86a47082/234.jpg"

alphabet_dict = {0: 'א', 1: 'ב', 2: 'ג', 3: 'ד', 4: 'ה', 5: 'ו', 6: 'ז', 7: 'ח', 8: 'ט',
                 9: 'י', 10: 'כ', 11: 'ן', 12: 'ל', 13: 'מ', 14: 'ם', 15: 'נ', 16: 'ן', 17: 'ס', 18: 'ע', 19: 'פ', 20: 'ף', 21: 'צ', 22: 'ץ', 23: 'ק', 24: 'ר', 25: 'ש', 26: 'ת'}

img_height = 28
img_width = 28

# pre-processing the image:
image = tf.keras.preprocessing.image.load_img(
    img_path, color_mode="grayscale", target_size=(img_height, img_width))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])

# predict
predictions = model.predict(input_arr)
class_index = np.argmax(predictions)
prediction_value = np.max(predictions)

# print predictions results
print(predictions)
print("the class index is: " + str(class_index) +
      "     confidence level: " + str(prediction_value))

# visualize prediction
font = {'family': 'sans-serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 18,
        }

plt.imshow(image)
plt.title('Predicted letter: {}'.format(
    alphabet_dict[class_index]), fontdict=font)
plt.axis('off')
plt.show()
