from ML_Module.classifier import Classifier
from pickle import load
from pickle import dump


class SKLearnModel(Classifier):
    """
    Class representing SKLearnModels

    """

    def __init__(self, classifier):
        self.classifier = classifier

    def __create_model(self):
        """
        Method used only at CNN

        """

        pass

    def train(self,
              training_set,
              labels):
        """
        :param training_set: graphs' feature vectors
        :param labels: graphs' labels
        """

        self.classifier.fit(training_set, labels)

    def predict_class(self,
                      test_set):
        """
        :param test_set: feature vectors for which to predict the labels
        :return: predicted labels
        """

        return self.classifier.predict(test_set)

    def predict_probs(self,
                      test_set):
        """
        :param test_set: feature vectors for which to predict the labels
        :return: predicted probabilities
        """

        return self.classifier.predict_proba(test_set)

    def save_model(self,
                   model_path):
        """
        :param model_path: path where to save the sklearn trained model
        """

        dump(self.classifier, open(model_path, 'wb'))

    def load_model(self,
                   model_path):
        """
        :param model_path: path from where to load the sklearn trained model
        """

        self.classifier = load(model_path, 'rb')
