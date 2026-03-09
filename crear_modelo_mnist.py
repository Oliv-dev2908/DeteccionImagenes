import tensorflow as tf
from tensorflow.keras import layers, models

# cargar dataset MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# normalizar
x_train = x_train / 255.0
x_test = x_test / 255.0

# reshape
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

# modelo simple
modelo = models.Sequential([
    layers.Conv2D(32,(3,3),activation="relu",input_shape=(28,28,1)),
    layers.MaxPooling2D(),

    layers.Conv2D(64,(3,3),activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128,activation="relu"),
    layers.Dense(10,activation="softmax")
])

modelo.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Entrenando modelo MNIST...")

modelo.fit(x_train, y_train, epochs=3)

modelo.save("mnist_model.h5")

print("Modelo guardado como mnist_model.h5")