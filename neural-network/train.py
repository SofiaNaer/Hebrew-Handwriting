import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import regularizers

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

img_height = 28
img_width = 28
batch_size = 32

model = keras.Sequential([
    layers.Input((28, 28, 1), name='image'),
    layers.RandomRotation(factor=0.03, fill_mode='reflect', interpolation='bilinear'),
    layers.Conv2D(64, 3, padding="same", activation='relu', name='conv1'),
    layers.BatchNormalization(),
    layers.Conv2D(64, 3, padding="same", activation='relu', name='conv2'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, 3, padding="same", activation='relu', name='conv3'),
    layers.BatchNormalization(),
    layers.Conv2D(128, 3, padding="same", activation='relu', name='conv4'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu', name='dense1'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu', name='dense2'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(27, activation='softmax', name='output')
])

#
# model = keras.Sequential(
#     [
#         layers.Input((28, 28, 1), name='image'),
#         layers.RandomRotation(factor=0.03, fill_mode='reflect', interpolation='bilinear'),
#         layers.Conv2D(64, 3, padding="same", activation='relu', name='conv1'),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Conv2D(128, 3, padding="same", activation='relu', name='conv2'),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Flatten(),
#         layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01), name='dense1'),
#         layers.Dropout(0.2),
#         layers.Dense(27, activation='softmax', kernel_regularizer=regularizers.l2(0.01), name='dense2'),
#     ]
# )
#
# model = keras.Sequential(
#     [
#         layers.Input((img_height, img_width, 1), name='image'),
#         layers.RandomRotation(factor=0.03, fill_mode='reflect', interpolation='bilinear'),
#         layers.Conv2D(128, 3, padding="same", activation='relu', name='conv1'),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Conv2D(256, 3, padding="same", activation='relu', name='conv2'),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Conv2D(256, 3, padding="same", activation='relu', name='conv3'),
#         layers.BatchNormalization(),
#         layers.Flatten(),
#         layers.Dense(128, activation='relu', name='dense1'),
#         layers.Dropout(0.5),
#         layers.Dense(27, activation='softmax', name='dense2'),
#     ]
# )

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    "../dataset",
    labels="inferred",
    label_mode='int',
    color_mode="grayscale",
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=123,
    validation_split=0.15,
    subset="training",
)

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    "../dataset",
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="validation",
)

ds_test = tf.keras.preprocessing.image_dataset_from_directory(
    "../dataset",
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="validation"
)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[keras.losses.SparseCategoricalCrossentropy()],
    metrics=["accuracy"])

print("\nFit model on training data:")
model.fit(ds_train, epochs=15, verbose=2, validation_data=ds_validation)
model.summary()

test_metrics = model.evaluate(ds_test, verbose=2)
print("\nTest Metrics:", test_metrics)

model.save("saved_model")
