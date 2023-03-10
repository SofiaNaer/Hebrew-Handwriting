# Imports needed
import tensorflow as tf
from tensorflow import keras
from keras import layers
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

img_height = 28
img_width = 28
batch_size = 16

model = keras.Sequential(
    [
        layers.Input((28, 28, 1)),
        layers.Conv2D(16, 3, padding="same"),
        layers.Conv2D(32, 3, padding="same"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(27),
    ]
)

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    labels="inferred",
    label_mode='int',  # categorical, binary
    color_mode="grayscale",
    batch_size=batch_size,
    image_size=(img_height, img_width),  # reshape if not in this size
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="training",
)

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    labels="inferred",
    label_mode="int",  # categorical, binary
    color_mode="grayscale",
    batch_size=batch_size,
    image_size=(img_height, img_width),  # reshape if not in this size
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="validation",
)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True),],
    metrics=["accuracy"],
)

print("\nFit model on training data:")
model.fit(ds_train, epochs=30, verbose=2)
model.summary()

print("\nEvaluate on test data:")
model.evaluate(ds_validation, verbose=2)

model.save("saved_model/")
