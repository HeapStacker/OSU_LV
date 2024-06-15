import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow import keras

fashion_data = tf.keras.datasets.mnist

# radi i train test split
(X_train, y_train), (X_test, y_test) = fashion_data.load_data() 

# print(X_train.shape) # (60000, 28, 28)
# print(X_test.shape) # (10000, 28, 28)

# trebamo preprocesirati podatke (pripremiti ih za neuralnu mrežu...)
X_train, X_test = X_train / 255, X_test / 255


model = keras.models.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=5)

model.evaluate(X_test, y_test)