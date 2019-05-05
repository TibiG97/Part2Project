from ML_Module.classifier import Classifier

from abc import abstractmethod

from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model


class NeuralNetwork(Classifier):
    """
    Class representing Neural Networks

    """

    def __init__(self,
                 classifier,
                 process_data,
                 batch_size: int,
                 epochs: int,
                 learning_rate: float,
                 dropout_rate: float,
                 init_mode: str,
                 no_of_classes: int,
                 verbose: int,
                 name: str):

        self.classifier = classifier
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.init_mode = init_mode
        self.no_of_classes = no_of_classes
        self.verbose = verbose

        super(NeuralNetwork, self).__init__(
            classifier=classifier,
            process_data=process_data,
            no_of_classes=no_of_classes,
            name=name
        )

    @abstractmethod
    def __create_model(self):
        pass

    def save_model(self,
                   model_path: str):
        """
        :param model_path: path where to save the model
        """

        self.classifier.model.save(model_path)

    def load_model(self,
                   model_path: str,
                   model_type: str):
        """
        :param model_path: path from where to load the model
        :param model_type: binary or multiclass
        """

        model = load_model(model_path)

        if model_type == 'binary':
            model.compile(loss="binary_crossentropy",
                          optimizer="adam",
                          metrics=["accuracy"])

        elif model_type == 'multiclass':
            model.compile(loss="categorical_crossentropy",
                          optimizer="adam",
                          metrics=["accuracy"])

        self.classifier = KerasClassifier(build_fn=model)
