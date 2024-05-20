import numpy as np
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# prikaz karakteristika train i test podataka
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# TODO: prikazi nekoliko slika iz train skupa
plt.imshow(x_train[0], cmap='gray')
plt.title("Category: " + str(y_train[0]))
plt.figure()
plt.imshow(x_train[1], cmap='gray')
plt.title("Category: " + str(y_train[1]))
plt.figure()
plt.imshow(x_train[2], cmap='gray')
plt.title("Category: " + str(y_train[2]))


# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")


# pretvori labele
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)


# TODO: kreiraj model pomocu keras.Sequential(); prikazi njegovu strukturu

# pretvori sliku iz 28x28 matricu u 784 elementni niz
x_train = x_train_s.reshape(60000, 784)
x_test = x_test_s.reshape(10000, 784)


model = keras.Sequential()
model.add(layers.Input(shape = (784, )))
model.add(layers.Dense(100, activation ="relu"))
model.add(layers.Dense(50, activation ="relu"))
model.add(layers.Dense(10, activation = "softmax"))
print(model.summary())

# TODO: definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=["accuracy",])


# TODO: provedi ucenje mreze
model.fit(x_train, y_train_s, batch_size=64, epochs=60, validation_split=0.1)


# TODO: Prikazi test accuracy i matricu zabune
score = model.evaluate( x_test, y_test_s, verbose =0)
print("Accuracy: ", score[1])
print("Loss: ", score[0])
print("Confusion matrix: \n", confusion_matrix(y_test, np.argmax(model.predict(x_test), axis=1)))


# TODO: spremi model
model.save("LV8_model.keras")

predicted = model.predict(x_test)

plt.show()