import numpy as np
import math
from ML_Module.CNN import CNN


def split_in_folds(data: list,
                   no_of_splits: int):
    """
    Private method that splits a data list into a specified number of folds

    :param data: dataset to be splitted
    :param no_of_splits: specified number of splits
    :return: A list of required number of sublists representing the mentioned splits
    """
    splits = list()
    size = len(data)
    for iterator in range(0, no_of_splits):
        current_split = list()
        left = int(size / no_of_splits) * iterator
        right = int(size / no_of_splits) * (iterator + 1)
        for index in range(left, right):
            current_split.append(data[index])
        splits.append(current_split)
    return splits


def merge_splits(data: list):
    """
    Private method that merges a list of lists representing the folds

    :param data: list of lists to be merged
    :return: a single list containing all elements in all folds
    """
    merged_list = list()
    for split in data:
        for element in split:
            merged_list.append(element)
    return merged_list


def accuracy(tp: int,
             tn: int,
             fp: int,
             fn: int):
    return (tp + tn) / (tp + tn + fp + fn)


def precision(tp: int,
              tn: int,
              fp: int,
              fn: int):
    return tp / (tp + fp)


def recall(tp: int,
           tn: int,
           fp: int,
           fn: int):
    return tp / (tp + fn)


def f1_score(tp: int,
             tn: int,
             fp: int,
             fn: int):
    return 2 * tp / (2 * tp + fp + fn)


def mcc(tp: int,
        tn: int,
        fp: int,
        fn: int):
    return (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))


def hyper_parameter_tuning(dataset: list,
                           parameters: list):
    return 0


def compute_metrics(predictions: list,
                    labels: list,
                    no_of_classes: int):
    """
    Private method computing the micro and macro evaluation metrics relevant for determining pipeline performance

    :param predictions: class predictions of a ML model
    :param labels: list of correct class labels
    :param no_of_classes: number of classes
    :return: evaluation metrics for (multiclass) classification
    """

    TP = list()  # true positives
    TN = list()  # true negatives
    FP = list()  # false positives
    FN = list()  # false negatives

    for positive_class in [-1, 1]:
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

    micro_accuracy = accuracy(sum(TP), sum(TN), sum(FP), sum(FN))
    micro_precision = precision(sum(TP), sum(TN), sum(FP), sum(FN))
    micro_recall = recall(sum(TP), sum(TN), sum(FP), sum(FN))
    micro_f1_score = f1_score(sum(TP), sum(TN), sum(FP), sum(FN))
    micro_mcc = mcc(sum(TP), sum(TN), sum(FP), sum(FN))

    micros = [micro_accuracy, micro_precision, micro_recall, micro_f1_score, micro_mcc]

    '''
    macro_accuracy = (1 / no_of_classes) * sum(accuracy(tp, tn, fp, fn) for (tp, tn, fp, fn) in (TP, TN, FP, FN))
    macro_precision = (1 / no_of_classes) * sum(precision(tp, tn, fp, fn) for (tp, tn, fp, fn) in (TP, TN, FP, FN))
    macro_recall = (1 / no_of_classes) * sum(recall(tp, tn, fp, fn) for (tp, tn, fp, fn) in (TP, TN, FP, FN))
    macro_f1_score = (1 / no_of_classes) * sum(f1_score(tp, tn, fp, fn) for (tp, tn, fp, fn) in (TP, TN, FP, FN))
    macro_mcc = (1 / no_of_classes) * sum(mcc(tp, tn, fp, fn) for (tp, tn, fp, fn) in (TP, TN, FP, FN))

    macros = [macro_accuracy, macro_precision, macro_recall, macro_f1_score, macro_mcc]
    '''

    return micros


def nested_cross_validation(data_set: list,
                            labels: list,
                            no_of_outer_folds: int,
                            no_of_inner_folds: int):
    file = open('/home/tiberiu/PycharmProjects/Part2Project/Data_Processing/random_stuff_file.txt', 'a')
    file.truncate(0)

    splitted_data_set = split_in_folds(data_set, no_of_outer_folds)
    splitted_labels = split_in_folds(labels, no_of_outer_folds)

    for outer_iterator in range(0, no_of_outer_folds):
        test_set = splitted_data_set[outer_iterator]
        test_labels = splitted_labels[outer_iterator]

        training_set = list()
        training_labels = list()
        for iterator in range(0, no_of_outer_folds):
            if iterator != outer_iterator:
                training_set.append(splitted_data_set[iterator])
                training_labels.append(splitted_labels[iterator])
        training_set = merge_splits(training_set)
        training_labels = merge_splits(training_labels)

        parameters = list()
        parameters = hyper_parameter_tuning(training_set, parameters)

        dummy = list()
        for iterator in range(0, 30):
            dummy.append(-1)

        cnn = CNN(w=10, k=2, epochs=2, batch_size=32,
                  verbose=2, attr_dim=30, dummy_value=dummy)

        cnn.fit(training_set, training_labels)
        predictions = cnn.predict(test_set)
        predictions = merge_splits(predictions)

        print(predictions)
        print(test_labels)
        print(compute_metrics(predictions, test_labels, 2), file=file)
