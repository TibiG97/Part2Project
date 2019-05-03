from ML_Module.classifier import Classifier
from pickle import load
from pickle import dump
import numpy as np


class SKLearnModel(Classifier):
    """
    Class representing SKLearnModels

    """

    def __init__(self,
                 classifier,
                 process_data,
                 name: str):
        self.classifier = classifier
        self.name = name

        super(SKLearnModel, self).__init__(
            classifier=classifier,
            process_data=process_data,
            name=name
        )

    def __process_data(self,
                       dataset: np.array):
        """
        Method used only at NeuralNet

        """

        pass

    def __create_model(self):
        """
        Method used only at NeuralNet

        """

        pass

    def save_model(self,
                   model_path):
        """
        :param model_path: path where to save the sklearn trained model
        """

        dump(self.classifier, open(model_path, 'wb'))

    def load_model(self,
                   model_path: str,
                   model_type: str):
        """
        :param model_path: path from where to load the sklearn trained model
        :param model_type: not used here
        """

        self.classifier = load(model_path, 'rb')
