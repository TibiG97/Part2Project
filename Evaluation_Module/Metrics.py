from math import sqrt
from sklearn.metrics import confusion_matrix


def accuracy(tp: int,
             tn: int,
             fp: int,
             fn: int):
    """
    Method that computed the accuracy of a ML model

    :param tp: true positives
    :param tn: true negatives
    :param fp: false positives
    :param fn: false negatives
    :return: accuracy score
    """
    if tp + tn == 0:
        return 0
    return (tp + tn) / (tp + tn + fp + fn)


def precision(tp: int,
              tn: int,
              fp: int,
              fn: int):
    """
    Method that computed the precision of a ML model

    :param tp: true positives
    :param tn: true negatives
    :param fp: false positives
    :param fn: false negatives
    :return: precision score
    """
    if tp == 0:
        return 0
    return tp / (tp + fp)


def recall(tp: int,
           tn: int,
           fp: int,
           fn: int):
    """
    Method that computed the recall of a ML model

    :param tp: true positives
    :param tn: true negatives
    :param fp: false positives
    :param fn: false negatives
    :return: recall score
    """
    if tp == 0:
        return 0
    return tp / (tp + fn)


def f1_score(tp: int,
             tn: int,
             fp: int,
             fn: int):
    """
    Method that computed the f1_score of a ML model

    :param tp: true positives
    :param tn: true negatives
    :param fp: false positives
    :param fn: false negatives
    :return: f1_score score
    """
    if tp == 0:
        return 0
    return 2 * tp / (2 * tp + fp + fn)


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
    metrics['accuracy'] = accuracy(tp, tn, fp, fn)
    metrics['precision'] = precision(tp, tn, fp, fn)
    metrics['recall'] = recall(tp, tn, fp, fn)
    metrics['f1_score'] = f1_score(tp, tn, fp, fn)
    metrics['mcc'] = mcc(tp, tn, fp, fn)

    return metrics


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

    TP = list()  # true positives
    TN = list()  # true negatives
    FP = list()  # false positives
    FN = list()  # false negatives

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

        TP.append(tp)
        TN.append(tn)
        FP.append(fp)
        FN.append(fn)

    # compute micro metrics
    #######################
    micro_accuracy = accuracy(sum(TP), sum(TN), sum(FP), sum(FN))
    micro_precision = precision(sum(TP), sum(TN), sum(FP), sum(FN))
    micro_recall = recall(sum(TP), sum(TN), sum(FP), sum(FN))
    micro_f1_score = f1_score(sum(TP), sum(TN), sum(FP), sum(FN))
    micro_mcc = mcc(sum(TP), sum(TN), sum(FP), sum(FN))

    micros = dict()
    micros['macro_accuracy'] = micro_accuracy
    micros['macro_precision'] = micro_precision
    micros['macro_recall'] = micro_recall
    micros['macro_f1_score'] = micro_f1_score
    micros['macro_mcc'] = micro_mcc
    #######################

    # compute macro metrics
    #######################
    macro_accuracy = (1 / no_of_classes) * sum(accuracy(tp, tn, fp, fn) for (tp, tn, fp, fn) in zip(TP, TN, FP, FN))
    macro_precision = (1 / no_of_classes) * sum(precision(tp, tn, fp, fn) for (tp, tn, fp, fn) in zip(TP, TN, FP, FN))
    macro_recall = (1 / no_of_classes) * sum(recall(tp, tn, fp, fn) for (tp, tn, fp, fn) in zip(TP, TN, FP, FN))
    macro_f1_score = (1 / no_of_classes) * sum(f1_score(tp, tn, fp, fn) for (tp, tn, fp, fn) in zip(TP, TN, FP, FN))
    macro_mcc = (1 / no_of_classes) * sum(mcc(tp, tn, fp, fn) for (tp, tn, fp, fn) in zip(TP, TN, FP, FN))

    macros = dict()
    macros['macro_accuracy'] = macro_accuracy
    macros['macro_precision'] = macro_precision
    macros['macro_recall'] = macro_recall
    macros['macro_f1_score'] = macro_f1_score
    macros['macro_mcc'] = macro_mcc
    #######################

    return micros, macros, confusion_matrix(predictions, labels)
