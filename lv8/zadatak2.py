import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt

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


# prikaz lose klasificiranih slika
predicted = model.predict(x_test)
predicted = np.argmax(predicted, axis=1)
# example values
#print("Predicted:\n" + str(predicted, axis=1)))
#print("True:\n" + str(y_test[0:10]))

wrong_predictions = predicted != y_test
wrong_images = x_test_s[wrong_predictions]
true_labels = y_test[wrong_predictions]
predicted_labels = predicted[wrong_predictions]

for i in range(3): # show first x wrong images
    plt.figure()
    plt.imshow(wrong_images[i], cmap='gray')
    plt.title("True: " + str(true_labels[i]) + " Predicted: " + str(predicted_labels[i]))



plt.show()