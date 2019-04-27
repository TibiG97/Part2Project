from ML_Module.CNN import CNN
from Evaluation_Module.Metrics import compute_metrics
from utils import merge_splits, split_in_folds, randomise_order


def hyper_parameter_tuning(dataset: list,
                           parameters: list):
    return 0


def nested_cross_validation(data_set: list,
                            labels: list,
                            no_of_outer_folds: int,
                            no_of_inner_folds: int):
    file = open('/home/tiberiu/PycharmProjects/Part2Project/results.txt', 'a')
    file.truncate(0)

    data_set, labels = randomise_order(data_set, labels)

    splitted_data_set = split_in_folds(data_set, no_of_outer_folds)
    splitted_labels = split_in_folds(labels, no_of_outer_folds)
    all_predictions = list()
    all_accuracies = list()

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

        cnn = CNN(w=10, k=2, epochs=10, batch_size=32,
                  verbose=2, attr_dim=30, dummy_value=dummy, multiclass=2)

        cnn.fit(training_set, training_labels)
        predictions = cnn.predict(test_set, test_labels)
        all_predictions.append(predictions)
        metrics = compute_metrics(predictions, test_labels, 2)
        print(metrics, file=file)
