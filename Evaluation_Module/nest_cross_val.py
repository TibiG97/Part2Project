from ML_Module.cnn import ConvolutionalNeuralNetwork
from Evaluation_Module.metrics import compute_metrics
from utils import merge_splits, split_in_folds, randomise_order
from sklearn.model_selection import cross_val_score
from utils import get_directory


def hyper_parameter_tuning(dataset: list,
                           parameters: list):
    return 0


def nested_cross_validation(data_set: list,
                            labels: list,
                            no_of_classes: int,
                            no_of_outer_folds: int,
                            no_of_inner_folds: int):
    file = open(get_directory() + '/results', 'a')
    file.truncate(0)

    data_set, labels = randomise_order(data_set, labels)

    splitted_data_set = split_in_folds(data_set, no_of_outer_folds)
    splitted_labels = split_in_folds(labels, no_of_outer_folds)
    all_predictions = list()
    all_accuracies = list()

    for outer_iterator in range(0, 10):
        print('ITERATION' + ' ' + str(outer_iterator), file=file)
        file.flush()
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

        cnn = ConvolutionalNeuralNetwork(w=10, k=2, epochs=30, batch_size=32,
                                         verbose=2, attr_dim=30, dummy_value=dummy, no_of_classes=no_of_classes)

        # best_model = cnn.cross_validate(training_set, training_labels, 10)

        # import numpy as np
        # predictions = best_model.predict(np.array(test_set))

        cnn.train(training_set, training_labels)
        predictions = cnn.predict_class(test_set)
        all_predictions.append(predictions)
        metrics = compute_metrics(predictions, test_labels, no_of_classes)
        for element in metrics:
            print(element, file=file)

        print(file=file)
        print(file=file)
