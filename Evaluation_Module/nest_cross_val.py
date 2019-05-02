from ML_Module.cnn import ConvolutionalNeuralNetwork
from ML_Module.mlp import MultilayerPerceptron

from ML_Module.baseline_models import RandomForest
from ML_Module.baseline_models import KNeighbours
from ML_Module.baseline_models import LogRegression

from Evaluation_Module.metrics import compute_metrics

from utils import merge_splits
from utils import split_in_folds
from utils import randomise_order
from utils import get_directory

from constants import DUMMY
from constants import CNN_GRID
from constants import MLP_GRID
from constants import RF_GRID
from constants import KNN_GRID
from constants import LOG_REG_GRID
from constants import ATTR_DIM

from copy import copy
import numpy as np

from random import randint


def tune_cnn_parameters(data_set: np.array,
                        labels: np.array,
                        no_of_folds: int,
                        no_of_samples: int,
                        no_of_classes: int):
    """
    Function that tunes the hyperparameter of the CNN model

    :return: best model found for each inner split, as well as the hyperparameters for which they were achieved
    """

    param_grid = copy(CNN_GRID)
    best_choices = None
    best_accuracy = 0

    for iterator in range(0, no_of_samples):

        choices = dict()
        for entry in param_grid.keys():
            position = randint(0, len(param_grid[entry]) - 1)
            choices[entry] = param_grid[entry][position]

        model = ConvolutionalNeuralNetwork(width=choices['width'],
                                           stride=choices['stride'],
                                           rf_size=choices['rf_size'],
                                           batch_size=choices['batch_size'],
                                           epochs=choices['epochs'],
                                           learning_rate=choices['learning_rate'],
                                           dropout_rate=choices['dropout_rate'],
                                           init_mode=choices['init_mode'],
                                           attr_dim=ATTR_DIM,
                                           dummy_value=DUMMY,
                                           no_of_classes=no_of_classes,
                                           verbose=2)

        average_acc = model.cross_validate(data_set, labels, no_of_folds)
        if average_acc / no_of_folds > best_accuracy:
            best_accuracy = average_acc / no_of_folds
            best_choices = choices

    # Replace the already trained model with a fresh one

    best_model = ConvolutionalNeuralNetwork(width=best_choices['width'],
                                            stride=best_choices['stride'],
                                            rf_size=best_choices['rf_size'],
                                            batch_size=best_choices['batch_size'],
                                            epochs=best_choices['epochs'],
                                            learning_rate=best_choices['learning_rate'],
                                            dropout_rate=best_choices['dropout_rate'],
                                            init_mode=best_choices['init_mode'],
                                            attr_dim=ATTR_DIM,
                                            dummy_value=DUMMY,
                                            no_of_classes=no_of_classes,
                                            verbose=2)

    return best_model, best_choices


def tune_mlp_parameters(data_set: np.array,
                        labels: np.array,
                        no_of_folds: int,
                        no_of_samples: int,
                        no_of_classes: int):
    """
    Function that tunes the hyperparameter of the MLP model

    :return: best model found for each inner split, as well as the hyperparameters for which they were achieved
    """

    param_grid = copy(MLP_GRID)
    best_choices = None
    best_accuracy = 0

    for iterator in range(0, no_of_samples):

        choices = dict()
        for entry in param_grid.keys():
            position = randint(0, len(param_grid[entry]) - 1)
            choices[entry] = param_grid[entry][position]

        print(choices)
        model = MultilayerPerceptron(hidden_size=choices['hidden_size'],
                                     batch_size=choices['batch_size'],
                                     epochs=choices['epochs'],
                                     learning_rate=choices['learning_rate'],
                                     dropout_rate=choices['dropout_rate'],
                                     init_mode=choices['init_mode'],
                                     no_of_classes=no_of_classes,
                                     verbose=2)

        average_acc = model.cross_validate(data_set, labels, no_of_folds)
        if average_acc / no_of_folds > best_accuracy:
            best_accuracy = average_acc / no_of_folds
            best_choices = choices

    # Replace the already trained model with a fresh one

    best_model = MultilayerPerceptron(hidden_size=best_choices['hidden_size'],
                                      batch_size=best_choices['batch_size'],
                                      epochs=best_choices['epochs'],
                                      learning_rate=best_choices['learning_rate'],
                                      dropout_rate=best_choices['dropout_rate'],
                                      init_mode=best_choices['init_mode'],
                                      no_of_classes=no_of_classes,
                                      verbose=2)

    return best_model, best_choices


def tune_rf_parameters(data_set: np.array,
                       labels: np.array,
                       no_of_folds: int,
                       no_of_samples: int):
    """
    Function that tunes the hyperparameter of the RandomForest model

    :return: best model found for each inner split, as well as the hyperparameters for which they were achieved
    """

    param_grid = copy(RF_GRID)
    best_choices = None
    best_accuracy = 0

    for iterator in range(0, no_of_samples):
        print(iterator)
        choices = dict()
        for entry in param_grid.keys():
            position = randint(0, len(param_grid[entry]) - 1)
            choices[entry] = param_grid[entry][position]

        model = RandomForest(depth=choices['depth'],
                             estimators=choices['estimators'],
                             samples_split=choices['samples_split'],
                             samples_leaf=choices['samples_leaf'])

        average_acc = model.cross_validate(data_set, labels, no_of_folds)
        if average_acc / no_of_folds > best_accuracy:
            best_accuracy = average_acc / no_of_folds
            best_choices = choices

    # Replace the already trained model with a fresh one

    best_model = RandomForest(depth=best_choices['depth'],
                              estimators=best_choices['estimators'],
                              samples_split=best_choices['samples_split'],
                              samples_leaf=best_choices['samples_leaf'])

    return best_model, best_choices


def tune_knn_parameters(data_set: np.array,
                        labels: np.array,
                        no_of_folds: int,
                        no_of_samples: int):
    """
    Function that tunes the hyperparameter of the KNeighbours model

    :return: best model found for each inner split, as well as the hyperparameters for which they were achieved
    """

    param_grid = copy(KNN_GRID)
    best_choices = None
    best_accuracy = 0

    for iterator in range(0, no_of_samples):
        print(iterator)
        choices = dict()
        for entry in param_grid.keys():
            position = randint(0, len(param_grid[entry]) - 1)
            choices[entry] = param_grid[entry][position]

        model = KNeighbours(neighbours=choices['neighbours'],
                            p_dist=choices['p_dist'])

        average_acc = model.cross_validate(data_set, labels, no_of_folds)
        if average_acc / no_of_folds > best_accuracy:
            best_accuracy = average_acc / no_of_folds
            best_choices = choices

    # Replace the already trained model with a fresh one

    best_model = KNeighbours(neighbours=best_choices['neighbours'],
                             p_dist=best_choices['p_dist'])

    return best_model, best_choices


def tune_lrg_parameters(data_set: np.array,
                        labels: np.array,
                        no_of_folds: int,
                        no_of_samples: int):
    """
    Function that tunes the hyperparameter of the LogRegression model

    :return: best model found for each inner split, as well as the hyperparameters for which they were achieved
    """

    param_grid = copy(LOG_REG_GRID)
    best_choices = None
    best_accuracy = 0

    for iterator in range(0, no_of_samples):
        print(iterator)
        choices = dict()
        for entry in param_grid.keys():
            position = randint(0, len(param_grid[entry]) - 1)
            choices[entry] = param_grid[entry][position]

        model = LogRegression(c=choices['c'],
                              penalty=choices['penalty'])

        average_acc = model.cross_validate(data_set, labels, no_of_folds)
        if average_acc / no_of_folds > best_accuracy:
            best_accuracy = average_acc / no_of_folds
            best_choices = choices

    # Replace the already trained model with a fresh one

    best_model = LogRegression(c=best_choices['c'],
                               penalty=best_choices['penalty'])

    return best_model, best_choices


def nested_cross_validation(data_set: np.array,
                            labels: np.array,
                            model_name: str,
                            tune_parameters: bool,
                            no_of_classes: int,
                            no_of_outer_folds: int,
                            no_of_inner_folds: int,
                            no_of_samples: int):
    """
    Function that performs nested cross validation for a specified model on a given dataset

    :param data_set: graph or attributes list, depending on the model
    :param labels: true class labels
    :param model_name: name of the model to be evaluated
    :param tune_parameters: wether to tune parameters or just to compute an outer CV
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

    for outer_iterator in range(0, 1):
        print(outer_iterator)
        print('ITERATION' + ' ' + str(outer_iterator), file=results_file)
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

        best_model = None
        best_parameters = None
        if tune_parameters:
            if model_name == 'CNN':
                best_model, best_parameters = tune_cnn_parameters(data_set=training_set,
                                                                  labels=training_labels,
                                                                  no_of_folds=no_of_inner_folds,
                                                                  no_of_samples=no_of_samples,
                                                                  no_of_classes=no_of_classes)
            if model_name == 'MLP':
                best_model, best_parameters = tune_mlp_parameters(data_set=training_set,
                                                                  labels=training_labels,
                                                                  no_of_folds=no_of_inner_folds,
                                                                  no_of_samples=no_of_samples,
                                                                  no_of_classes=no_of_classes)
            if model_name == 'RF':
                best_model, best_parameters = tune_rf_parameters(data_set=training_set,
                                                                 labels=training_labels,
                                                                 no_of_folds=no_of_inner_folds,
                                                                 no_of_samples=no_of_samples)
            if model_name == 'KNN':
                best_model, best_parameters = tune_knn_parameters(data_set=training_set,
                                                                  labels=training_labels,
                                                                  no_of_folds=no_of_inner_folds,
                                                                  no_of_samples=no_of_samples)
            if model_name == 'LRG':
                best_model, best_parameters = tune_lrg_parameters(data_set=training_set,
                                                                  labels=training_labels,
                                                                  no_of_folds=no_of_inner_folds,
                                                                  no_of_samples=no_of_samples)

        else:
            best_model = ConvolutionalNeuralNetwork(width=20,
                                                    stride=1,
                                                    rf_size=20,
                                                    epochs=100,
                                                    batch_size=32,
                                                    learning_rate=0.001,
                                                    dropout_rate=0.5,
                                                    no_of_classes=no_of_classes)
            '''
            best_model = MultilayerPerceptron(epochs=50,
                                              batch_size=32,
                                              verbose=2,
                                              dropout_rate=0.5,
                                              hidden_size=128,
                                              init_mode='he',
                                              learning_rate=0.001,
                                              no_of_classes=no_of_classes)
            '''

        if best_parameters is not None:
            print(best_parameters, file=results_file)

        best_model.train(training_set, training_labels)
        predictions = best_model.predict_class(test_set)
        all_predictions.append(predictions)
        metrics = compute_metrics(predictions, test_labels, no_of_classes)

        for element in metrics:
            print(element, file=results_file)

        print(file=results_file)
        print(file=results_file)
