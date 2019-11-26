import matplotlib.pyplot as plt
from torch import from_numpy
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm
from sklearn import metrics


def error_plot(loss):
    plt.figure(figsize=(10, 5))
    plt.plot(loss)
    plt.title("loss plot")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.show()


def acc_plot(acc):
    plt.figure(figsize=(10, 5))
    plt.plot(acc)
    plt.title("accuracy plot")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """This function prints and plots the confusion matrix.
     
    Args:
        y_true: 2d numpy array of true labels, rows - objects, columns - attributes
        y_pred: 2d numpy array of predicted labels
        normalize:bool, if true - apply normalization 
        title: str
        cmap: plt.cm object, color map

    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    conf_mx = metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        conf_mx = conf_mx.astype('float') / conf_mx.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(conf_mx)

    fig, ax = plt.subplots()
    im = ax.imshow(conf_mx, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(conf_mx.shape[1]),
           yticks=np.arange(conf_mx.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = conf_mx.max() / 2.
    for i in range(conf_mx.shape[0]):
        for j in range(conf_mx.shape[1]):
            ax.text(j, i, format(conf_mx[i, j], fmt),
                    ha="center", va="center",
                    color="white" if conf_mx[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def performance_metrics(true, pred):
    precision = metrics.precision_score(true, pred, average='macro')
    recall = metrics.recall_score(true, pred, average='macro')
    accuracy = metrics.accuracy_score(true, pred)
    f1_score = metrics.f1_score(true, pred, average='macro')
    print('Precision: {} Recall: {}, Accuracy: {}: ,f1_score: {}'.format(
        precision*100, recall*100, accuracy*100, f1_score*100))