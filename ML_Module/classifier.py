from abc import abstractmethod
import numpy as np


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

        if self.name == 'MLP' or self.name == 'CNN':
            training_set = self.process_data(training_set)

        self.classifier.fit(training_set, labels)

    def predict_class(self,
                      test_set: np.array):
        """
        :param test_set: feature vectors for which to predict the labels
        :return: predicted labels
        """

        if self.name == 'MLP' or self.name == 'CNN':
            test_set = self.process_data(test_set)

        return self.classifier.predict(test_set)

    def predict_probs(self,
                      test_set: np.array):
        """
        :param test_set: feature vectors for which to predict the labels
        :return: predicted probabilities
        """

        if self.name == 'MLP' or self.name == 'CNN':
            test_set = self.process_data(test_set)

        return self.classifier.predict_proba(test_set)

    @abstractmethod
    def save_model(self,
                   model_path: str):
        pass

    @abstractmethod
    def load_model(self,
                   model_path: str,
                   model_type: str):
        pass
