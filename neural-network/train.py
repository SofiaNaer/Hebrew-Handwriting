import tensorflow as tf
from tensorflow import keras
from keras import layers
import os


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

img_height = 28
img_width = 28
batch_size = 32

model = keras.Sequential(
    [
        layers.Input((28, 28, 1), name='image'),
        layers.Conv2D(32, 3, padding="same", activation='relu', name='conv1'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, 3, padding="same", activation='relu', name='conv2'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu', name='dense1'),
        layers.Dropout(0.2),
        layers.Dense(27, activation='softmax', name='dense2'),
    ]
)

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    "../dataset",
    labels="inferred",
    label_mode='int',  # categorical, binary
    color_mode="grayscale",
    batch_size=batch_size,
    image_size=(img_height, img_width),  # reshape if not in this size
    shuffle=True,
    seed=123,
    validation_split=0.15,
    subset="training",
)

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    "../dataset",
    labels="inferred",
    label_mode="int",  # categorical, binary
    color_mode="grayscale",
    batch_size=batch_size,
    image_size=(img_height, img_width),  # reshape if not in this size
    shuffle=True,
    seed=123,
    validation_split=0.15,
    subset="validation",
)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[keras.losses.SparseCategoricalCrossentropy()],
    metrics=["accuracy"],
)

print("\nFit model on training data:")
model.fit(ds_train, epochs=40, verbose=2, validation_data=ds_validation)
model.summary()

print("\nEvaluate on validation data:")
model.evaluate(ds_validation, verbose=2)

model.save("saved_model")
