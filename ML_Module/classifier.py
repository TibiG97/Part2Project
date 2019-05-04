from abc import abstractmethod
import numpy as np

from utils import randomise_order
from utils import split_in_folds
from utils import merge_splits

from keras.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score


class Classifier(object):
    """
    Abstract class specifying the mandatory core functionality of all the classifiers implemented

    """

    def __init__(self,
                 classifier,
                 process_data,
                 name: str):
        self.classifier = classifier
        self.name = name
        self.process_data = process_data

    def train(self,
              training_set: np.array,
              labels: np.array):
        """
        :param training_set: graphs' feature vectors
        :param labels: graphs' labels
        """

        if self.name == 'CNN' or self.name == 'LRG':
            training_set = self.process_data(training_set)

        if self.name == 'CNN' or self.name == 'MLP':
            es = EarlyStopping(monitor='loss', mode='min', patience=10)
            self.classifier.fit(training_set, labels, callbacks=[es])
        else:
            self.classifier.fit(training_set, labels)

    def predict_class(self,
                      test_set: np.array):
        """
        :param test_set: feature vectors for which to predict the labels
        :return: predicted labels
        """

        if self.name == 'CNN' or self.name == 'LRG':
            test_set = self.process_data(test_set)

        return self.classifier.predict(test_set)

    def predict_probs(self,
                      test_set: np.array):
        """
        :param test_set: feature vectors for which to predict the labels
        :return: predicted probabilities
        """

        if self.name == 'CNN' or self.name == 'LRG':
            test_set = self.process_data(test_set)

        return self.classifier.predict_proba(test_set)

    def cross_validate(self,
                       data_set: np.array,
                       labels: np.array,
                       no_of_folds: int):
        """
        Method that performs a CV on the object modelon a given dataset

        :param data_set: graph or attributes list, depending on the model
        :param labels: true class labels
        :param no_of_folds: number of folds for the CV
        :return: average accuracy of the model on all splits
        """

        data_set, labels, permutation = randomise_order(data_set, labels)

        splitted_data_set = split_in_folds(data_set, no_of_folds)
        splitted_labels = split_in_folds(labels, no_of_folds)
        all_accuracies = list()
        average_acc = 0
        all_predictions = list()

        for index1 in range(0, no_of_folds):
            print("Inner Fold #" + str(index1 + 1))

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

            self.train(training_set, training_labels)
            predictions = self.predict_class(test_set)
            for prediction in predictions:
                all_predictions.append(prediction)

            acc = accuracy_score(test_labels, predictions)
            all_accuracies.append(acc)
            average_acc += acc

        average_acc = average_acc / no_of_folds
        permutation, all_predictions = (list(t) for t in zip(*sorted(zip(permutation, all_predictions))))

        return average_acc, all_accuracies, all_predictions

    @abstractmethod
    def save_model(self,
                   model_path: str):
        pass

    @abstractmethod
    def load_model(self,
                   model_path: str,
                   model_type: str):
        pass
