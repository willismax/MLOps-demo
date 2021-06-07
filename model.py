# Import modules and packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


print(tf.__version__)


# Data set: Fasfion_MNIST
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# Set random seed
tf.random.set_seed(42)

X_train = X_train / 255.0
X_test = X_test / 255.0

# Create a model using the Sequential API
model = tf.keras.Sequential([
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(500, activation='sigmoid'),
                tf.keras.layers.Dense(500, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
                ])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])


# Fit the model
model.fit(X_train, y_train, epochs=10)

# Make and plot predictions for model_1
y_preds = model.predict(X_test)

# Calculate model_1 metrics
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

print(f'\nTest accuracy: {test_acc}')
print(f'Test loss: {test_loss}')

# Write metrics to file
with open('result.txt', 'w') as outfile:
    outfile.write(f'\nTest accuracy: {test_acc}, Test loss: {test_loss}.')
