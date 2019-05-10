from abc import abstractmethod
import numpy as np

from utils import randomise_order
from utils import split_in_folds
from utils import merge_splits

from keras.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from Evaluation_Module.metrics import compute_metrics


class Classifier(object):
    """
    Abstract class specifying the mandatory core functionality of all the classifiers implemented

    """

    def __init__(self,
                 classifier,
                 process_data,
                 no_of_classes: int,
                 name: str):
        self.classifier = classifier
        self.name = name
        self.process_data = process_data
        self.no_of_classes = no_of_classes

    def train(self,
              training_set: np.array,
              labels: np.array):
        """
        :param training_set: graphs' feature vectors
        :param labels: graphs' labels
        """

        print(self.name + ' is now training')

        if self.name == 'CNN':
            training_set = self.process_data(training_set)

        if self.name == 'CNN' or self.name == 'MLP':
            es = EarlyStopping(monitor='val_loss', mode='min', patience=10)
            return self.classifier.fit(training_set, labels, callbacks=[es])
        else:
            self.classifier.fit(training_set, labels)

    def predict_class(self,
                      test_set: np.array):
        """
        :param test_set: feature vectors for which to predict the labels
        :return: predicted labels
        """
        print(self.name + ' is now predicting classes')

        if self.name == 'CNN':
            test_set = self.process_data(test_set)

        return self.classifier.predict(test_set)

    def predict_probs(self,
                      test_set: np.array):
        """
        :param test_set: feature vectors for which to predict the labels
        :return: predicted probabilities
        """
        print(self.name + ' is now predincting probabilities')

        if self.name == 'CNN':
            test_set = self.process_data(test_set)

        return self.classifier.predict_proba(test_set)

    def cross_validate(self,
                       data_set: np.array,
                       labels: np.array,
                       no_of_folds: int,
                       clear_file=False):
        """
        Method that performs a CV on the object modelon a given dataset

        :param data_set: graph or attributes list, depending on the model
        :param labels: true class labels
        :param no_of_folds: number of folds for the CV
        :param clear_file: used to clear file when running just a outer CV
        :return: average accuracy of the model on all splits
        """
        model_history = list()
        results_file = open('Results/' + self.name, 'a')
        if clear_file:
            results_file.truncate(0)

        data_set, labels, permutation = randomise_order(data_set, labels)

        splitted_data_set = split_in_folds(data_set, no_of_folds)
        splitted_labels = split_in_folds(labels, no_of_folds)
        all_accuracies = list()
        average_acc = 0
        all_predictions = list()

        for index1 in range(0, no_of_folds):
            print("Inner Fold #" + str(index1 + 1))
            print("Inner Fold #" + str(index1 + 1), file=results_file)
            print(file=results_file)

            test_set = splitted_data_set[index1]
            test_labels = splitted_labels[index1]

            training_set = list()
            training_labels = list()
            for index2 in range(0, no_of_folds):
                if index2 != index1:
                    training_set.append(splitted_data_set[index2])
                    training_labels.append(splitted_labels[index2])
            training_set = merge_splits(training_set)
            training_labels = merge_splits(training_labels)

            history = self.train(training_set, training_labels)
            model_history.append(history)

            predictions = self.predict_class(test_set)
            for prediction in predictions:
                all_predictions.append(prediction)

            metrics = compute_metrics(test_labels, predictions, self.no_of_classes)
            for element in metrics:
                print(element, file=results_file)
            print(file=results_file)
            print(file=results_file)
            print()

            acc = accuracy_score(test_labels, predictions)
            all_accuracies.append(acc)
            average_acc += acc

        print(confusion_matrix(labels, all_predictions), file=results_file)
        print(file=results_file)

        average_acc = average_acc / no_of_folds
        permutation, all_predictions = (list(t) for t in zip(*sorted(zip(permutation, all_predictions))))

        print('Average accuracy across ' + str(no_of_folds) + ' inner splits: ' + str(average_acc), file=results_file)
        print(file=results_file)

        return average_acc, all_accuracies, all_predictions, model_history

    @abstractmethod
    def save_model(self,
                   model_path: str):
        pass

    @abstractmethod
    def load_model(self,
                   model_path: str,
                   model_type: str):
        pass
