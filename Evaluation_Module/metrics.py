from math import sqrt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def mcc(tp: int,
        tn: int,
        fp: int,
        fn: int):
    """
    Method that computed the Matthew's Corelation Coefficient of a ML model

    :param tp: true positives
    :param tn: true negatives
    :param fp: false positives
    :param fn: false negatives
    :return: MCC score
    """

    if tp * tn - fp * fn == 0:
        return 0
    return (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))


def compute_binary_metrics(predictions: list,
                           labels: list):
    """
    Method computing the evaluation metrics relevant for determining pipeline performance on binary classification

    :param predictions: class predictions of a ML model
    :param labels: list of correct class labels
    :return: evaluation metrics for binary classification
    """

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    positive_class = 1

    for index in range(0, len(predictions)):
        if predictions[index] == positive_class and labels[index] == positive_class:
            tp += 1
        if predictions[index] == positive_class and labels[index] != positive_class:
            fp += 1
        if predictions[index] != positive_class and labels[index] == positive_class:
            fn += 1
        if predictions[index] != positive_class and labels[index] != positive_class:
            tn += 1

    metrics = dict()
    metrics['accuracy'] = accuracy_score(labels, predictions)
    metrics['precision'] = precision_score(labels, predictions)
    metrics['recall'] = recall_score(labels, predictions)
    metrics['f1_score'] = f1_score(labels, predictions)
    metrics['mcc'] = mcc(tp, tn, fp, fn)

    return metrics, confusion_matrix(labels, predictions)


def compute_metrics(predictions: list,
                    labels: list,
                    no_of_classes: int):
    """
    Method computing the micro and macro evaluation metrics relevant for determining pipeline performance

    :param predictions: class predictions of a ML model
    :param labels: list of correct class labels
    :param no_of_classes: number of classes
    :return: micro and macro evaluation metrics for (multiclass) classification
    """

    if no_of_classes == 2:
        return compute_binary_metrics(predictions, labels)

    tp_list = list()  # true positives
    tn_list = list()  # true negatives
    fp_list = list()  # false positives
    fn_list = list()  # false negatives

    for positive_class in range(1, no_of_classes + 1):
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for index in range(0, len(predictions)):
            if predictions[index] == positive_class and labels[index] == positive_class:
                tp += 1
            if predictions[index] == positive_class and labels[index] != positive_class:
                fp += 1
            if predictions[index] != positive_class and labels[index] == positive_class:
                fn += 1
            if predictions[index] != positive_class and labels[index] != positive_class:
                tn += 1

        tp_list.append(tp)
        tn_list.append(tn)
        fp_list.append(fp)
        fn_list.append(fn)

    # compute micro metrics
    #######################
    micro_accuracy = accuracy_score(labels, predictions)
    micro_precision = precision_score(labels, predictions, average='micro')
    micro_recall = recall_score(labels, predictions, average='micro')
    micro_f1_score = f1_score(labels, predictions, average='micro')
    micro_mcc = mcc(sum(tp_list), sum(tn_list), sum(fp_list), sum(fn_list))

    micros = dict()
    micros['micro_accuracy'] = round(micro_accuracy, 3)
    micros['micro_precision'] = round(micro_precision, 3)
    micros['micro_recall'] = round(micro_recall, 3)
    micros['micro_f1_score'] = round(micro_f1_score, 3)
    micros['micro_mcc'] = round(micro_mcc, 3)
    #######################

    # compute macro metrics
    #######################
    macro_accuracy = accuracy_score(labels, predictions)
    macro_precision = precision_score(labels, predictions, average='macro')
    macro_recall = recall_score(labels, predictions, average='macro')
    macro_f1_score = f1_score(labels, predictions, average='macro')
    macro_mcc = (1 / no_of_classes) * sum(
        mcc(tp, tn, fp, fn) for (tp, tn, fp, fn) in zip(tp_list, tn_list, fp_list, fn_list))

    macros = dict()
    macros['macro_accuracy'] = round(macro_accuracy, 3)
    macros['macro_precision'] = round(macro_precision, 3)
    macros['macro_recall'] = round(macro_recall, 3)
    macros['macro_f1_score'] = round(macro_f1_score, 3)
    macros['macro_mcc'] = round(macro_mcc, 3)
    #######################

    return micros, macros, confusion_matrix(predictions, labels)
