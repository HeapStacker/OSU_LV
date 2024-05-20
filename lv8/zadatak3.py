import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
from PIL import Image

model = keras.models.load_model("LV8_model.keras")

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)


# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

# pretvori sliku iz 28x28 matricu u 784 elementni niz
x_train = x_train_s.reshape(60000, 784)
x_test = x_test_s.reshape(10000, 784)


# pretvori labele
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)


def prepare_image(image):
    image = image.convert("L")  # grayscale
    image = image.resize((28, 28))  # resize image

    image = np.array(image)
    image = image.astype("float32") / 255
    image = np.expand_dims(image, axis=-1)
    image = image.reshape(1, 784)
    return image



# test the model on new images
images = ["test2.png", "test5.png", "test7.png", "test8.png", "example.png"]

for image in images:
    image_path = "lv8/test_images/" + image
    image = Image.open(image_path)

    image = prepare_image(image)

    # Make predictions
    predictions = model.predict(image)
    predicted_label = np.argmax(predictions[0])

    # Display the image and predicted label
    plt.figure()
    plt.imshow(image.reshape(28, 28), cmap="gray")
    plt.title(f"Predicted Label: {predicted_label}")


plt.show()