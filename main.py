import tensorflow as tf
print("TEsnorflow version:", tf.__version__)

# refers to the Keras Built-in dataset MNIST
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0