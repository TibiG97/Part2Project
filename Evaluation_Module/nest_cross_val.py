from ML_Module.baseline_models import LogRegression
from ML_Module.baseline_models import KNeighbours
from ML_Module.baseline_models import RandomForest

from ML_Module.mlp import MultilayerPerceptron
from ML_Module.cnn import ConvolutionalNeuralNetwork

from Evaluation_Module.metrics import compute_metrics

from utils import merge_splits
from utils import split_in_folds
from utils import randomise_order
from utils import get_directory

from constants import GRIDS

from copy import copy
import numpy as np

from random import randint

MODELS = {
    "CNN": ConvolutionalNeuralNetwork,
    "MLP": MultilayerPerceptron,
    "LRG": LogRegression,
    "KNN": KNeighbours,
    "RF": RandomForest
}


def hyperparameter_tuning(data_set: np.array,
                          labels: np.array,
                          no_of_classes: int,
                          no_of_folds: int,
                          no_of_samples: int,
                          model_name: str):
    """
    Method that performs hyperparameter tuning of a specific model on a given dataset.
    There are no_of_samples cross validated models on the dataset, obtained in RandomSearch style

    :return: best performing model (in terms of accuracy)
    """

    param_grid = copy(GRIDS[model_name])
    best_choices = None
    best_accuracy = 0

    for iterator in range(0, no_of_samples):

        print()
        print('Sample #' + str(iterator + 1))

        choices = dict()
        for entry in param_grid.keys():
            position = randint(0, len(param_grid[entry]) - 1)
            choices[entry] = param_grid[entry][position]

        model = MODELS[model_name](**choices,
                                   no_of_classes=no_of_classes,
                                   verbose=0)

        average_acc = model.cross_validate(data_set, labels, no_of_folds)
        if average_acc / no_of_folds > best_accuracy:
            best_accuracy = average_acc / no_of_folds
            best_choices = choices

    # Replace the already trained model with a fresh one
    best_model = MODELS[model_name](**best_choices,
                                    no_of_classes=no_of_classes,
                                    verbose=0)

    print()
    print()

    return best_model, best_choices


def nested_cross_validation(data_set: np.array,
                            labels: np.array,
                            model_name: str,
                            no_of_classes: int,
                            no_of_outer_folds: int,
                            no_of_inner_folds: int,
                            no_of_samples: int):
    """
    Method that performs nested cross validation for a specified model on a given dataset

    :param data_set: graph or attributes list, depending on the model
    :param labels: true class labels
    :param model_name: name of the model to be evaluated
    :param no_of_classes: number of classes in the dataset
    :param no_of_outer_folds: number of outer folds in the NCV
    :param no_of_inner_folds: number of inner folds in the NCV
    :param no_of_samples: number of samples to be generated in the RandomSearchCV
    """

    results_file = open(get_directory() + '/Results/' + model_name, 'a')
    results_file.truncate(0)

    data_set, labels = randomise_order(data_set, labels)

    splitted_data_set = split_in_folds(data_set, no_of_outer_folds)
    splitted_labels = split_in_folds(labels, no_of_outer_folds)
    all_predictions = list()

    for outer_iterator in range(0, no_of_outer_folds):
        print('Outer Fold #' + str(outer_iterator + 1))
        print('ITERATION' + ' ' + str(outer_iterator + 1), file=results_file)
        results_file.flush()

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

        best_model, best_parameters = hyperparameter_tuning(data_set=training_set,
                                                            labels=training_labels,
                                                            no_of_classes=no_of_classes,
                                                            no_of_folds=no_of_inner_folds,
                                                            no_of_samples=no_of_samples,
                                                            model_name=model_name)

        print(best_parameters, file=results_file)

        best_model.train(training_set, training_labels)
        predictions = best_model.predict_class(test_set)
        all_predictions.append(predictions)
        metrics = compute_metrics(predictions, test_labels, no_of_classes)

        for element in metrics:
            print(element, file=results_file)

        print(file=results_file)
        print(file=results_file)
