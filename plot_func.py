# Import modules and packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(y_true, y_pred, classes=None, normalize=False):
    """Compute confusion matrix with sklearn.metrics.confusion_matrix()
    
    Args:
        y_true: true lable.
        y_pred: predicted lable.
        classes: all kinds of classes, defult use 
            sklearn.utils.multiclass.unique_labels().
        normalize: normalized confusion matrix data.
    Returns:
        plot_confusion_matrix.png
    """
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
    """Plot training & validation accuracy values
    
    Args: 
        history: result of model.fit().
    Returns:
        plot_loss.png
    """
    plt.plot(history.history[accuracy_label])
    plt.plot(history.history['val_'+accuracy_label])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('plot_accuracy.png', dpi=120)
    plt.close()

def plot_loss(history):
    """Plot training & validation loss values
    
        Args: 
        history: result of model.fit().
    Returns:
        plot_loss.png
    """
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('plot_loss.png', dpi=120)
    plt.close()