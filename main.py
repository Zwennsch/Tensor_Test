import tensorflow as tf
print("Tensorflow version:", tf.__version__)

# refers to the Keras Built-in dataset MNIST
mnist = tf.keras.datasets.mnist

# load_data() returns a tuple of NumPy arrays
# load the dataset: x_train contains 60000 grayscale images of the digits 0-9. y_train contains the corresponding digit labels from 0-9. 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixel values (0-255 -> 0-1)
x_train, x_test = x_train / 255.0, x_test / 255.0


# Build a machine learning model. Sequential is useful for stacking layers where each layer has one input and one output tensor. 
# Layers are functions with a known mathematical structure that can be reused and have trainable variables. 
# Most tensorflow models are composed of layers. Here: Flatten, Dense and Dropout.
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten 28x28 images to 1D array
    tf.keras.layers.Dense(128, activation='relu'),  # Fully connected layer with ReLU
    tf.keras.layers.Dropout(0.2),                  # Prevent overfitting
    tf.keras.layers.Dense(10, activation='softmax') # Output layer (10 classes)
])

predictions = model(x_train[:1]).numpy()
print(predictions)