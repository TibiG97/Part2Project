from ML_Module.classifier import Classifier
from abc import abstractmethod


class NeuralNetwork(Classifier):
    """
    Class representing Neural Networks

    """

    @abstractmethod
    def __create_model(self):
        pass

    def train(self,
              training_set,
              labels):
        pass

    def predict_class(self,
                      test_set):
        pass

    def predict_probs(self,
                      test_set):
        pass

    def save_model(self,
                   model_path: str):
        pass

    def load_model(self,
                   model_path: str):
        pass

    @abstractmethod
    def __process_data(self):
        pass
