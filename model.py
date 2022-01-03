# Import modules and packages
from numpy.lib.npyio import save
import tensorflow as tf
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import numpy as np


print(tf.__version__)

# Functions and procedures
# def plot_predictions(train_data, train_labels,  test_data, test_labels,  predictions):
#     """
#     Plots training data, test data and compares predictions.
#     """
#     plt.figure(figsize=(6, 5))
#     # Plot training data in blue
#     plt.scatter(train_data, train_labels, c="b", label="Training data")
#     # Plot test data in green
#     plt.scatter(test_data, test_labels, c="g", label="Testing data")
#     # Plot the predictions in red (predictions were made on the test data)
#     plt.scatter(test_data, predictions, c="r", label="Predictions")
#     # Show the legend
#     plt.legend(shadow='True')
#     # Set grids
#     plt.grid(which='major', c='#cccccc', linestyle='--', alpha=0.5)
#     # Some text
#     plt.title('Model Results', family='Arial', fontsize=14)
#     plt.xlabel('X axis values', family='Arial', fontsize=11)
#     plt.ylabel('Y axis values', family='Arial', fontsize=11)
#     # Show
#     plt.savefig('model_results.png', dpi=120)

# def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(names))
#     plt.xticks(tick_marks, names, rotation=90)
#     plt.yticks(tick_marks, names)
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')



# Data set: Fasfion_MNIST
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# Set random seed
tf.random.set_seed(520)

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
history = model.fit(X_train, y_train, validation_split=0.25, epochs=8)

# Make predictions for model_1
y_preds = model.predict(X_test)

# Plot predictions for model_1
# plot_predictions(train_data=X_train[:,0][:,0], train_labels=y_train,  test_data=X_test, test_labels=y_test,  predictions=y_preds)

# Calculate model_1 metrics
test_loss, test_acc = model.evaluate(X_test, y_test)

print(f'\nTest accuracy: {test_acc}')
print(f'Test loss: {test_loss}')



def plot_confusion_matrix(y_true, y_pred, classes=None, normalize=False):
    # Compute confusion matrix with sklearn.metrics.confusion_matrix()
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    unique = unique_labels(y_true, y_pred)
    classes = unique if (classes is None) else classes[unique]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]), 
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes, 
        yticklabels=classes,
        title='Confusion matrix',
        ylabel='True label',
        xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(
        ax.get_xticklabels(), 
        rotation=45, 
        ha="right",
        rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, 
                i, 
                format(cm[i, j], fmt),
                ha="center", 
                va="center",
                color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    # Save confusion_matrix plot.
    plt.savefig('plot_confusion_matrix.png', dpi=120)
    plt.close()
    return ax

def plot_accuracy(history):
    accuracy_label = 'accuracy' if 'accuracy' in history.history else 'acc'
    """Plot training & validation accuracy values"""
    plt.plot(history.history[accuracy_label])
    plt.plot(history.history['val_'+accuracy_label])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('plot_accuracy.png', dpi=120)
    plt.close()

def plot_loss(history):
    """Plot training & validation loss values"""
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('plot_loss.png', dpi=120)
    plt.close()

predicted_classes = np.argmax(y_preds, axis = 1)
plot_confusion_matrix(y_pred=predicted_classes, y_true=y_test, classes=np.array(class_names))
plot_accuracy(history)
plot_loss(history)

# Write metrics to file
with open('result.txt', 'w') as outfile:
    outfile.write(f'\nTest accuracy: {test_acc}, Test loss: {test_loss}.')
