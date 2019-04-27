import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np


def plot_accuracy_vs_epoch(model):
    print(model.history.keys())
    plt.plot(model.history['acc'])
    # plt.plot(model.model['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    # plt.show()
    return


def plot_roc_curve(predict, y_test):
    y_test = np.array(y_test)
    fpr_keras, tpr_keras, threshold_keras = roc_curve(y_test, predict)
    auc_keras = auc(fpr_keras, tpr_keras)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    # plt.show()
    return


def plot_precision_recall_curve(probs, y_test):
    precision, recall, threshold = precision_recall_curve(y_test, probs)
    a = auc(recall, precision)
    plt.plot([0, 1], [0.5, 0.5], linestyle='--')
    plt.plot(recall, precision, marker='.')
    # plt.show()
    return


def plot_accuracy_per_fold(accuracies):
    n = len(accuracies[0])

    fig, ax = plt.subplots(figsize=(8, 8))
    ind = np.arange(n)  # the x locations for the groups
    print(ind)
    width = 0.18  # the width of the bars
    p1 = ax.bar(ind, accuracies[0], width, color='r', bottom=0)
    p2 = ax.bar(ind + width, accuracies[1], width, color='y', bottom=0)
    p3 = ax.bar(ind + 2 * width, accuracies[2], width, color='g', bottom=0)
    p4 = ax.bar(ind + 3 * width, accuracies[3], width, color='b', bottom=0)

    ax.set_xlabel('Fold_Number')
    ax.set_ylabel('Accuracies')
    ax.set_xticks(ind + width * 3 / 2)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1])
    ax.set_xticklabels(('F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9'))
    ax.legend((p1[0], p2[0], p3[0], p4[0]), ('Patchy-San CNN', 'KNeighbours', 'DecisionTree', 'RandomForest'),
              loc='upper right', ncol=2)

    plt.show()
    return
