from ML_Module.neural_net import NeuralNetwork

import os
import glob
import shutil
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from keras.preprocessing.text import Tokenizer
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam


class MultilayerPerceptron(NeuralNetwork):
    """
    Class representing Multilayer Peceptrons

    """

    def __init__(self,
                 batch_size: int,
                 epochs: int,
                 learning_rate: float,
                 dropout_rate: float,
                 init_mode: str,
                 no_of_classes: int,
                 verbose: int,
                 hidden_size: int):

        self.hidden_size = hidden_size
        self.classifier = KerasClassifier(build_fn=self.__create_model,
                                          epochs=epochs,
                                          batch_size=batch_size,
                                          verbose=verbose)

        super(MultilayerPerceptron, self).__init__(
            classifier=self.classifier,
            process_data=None,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            init_mode=init_mode,
            no_of_classes=no_of_classes,
            verbose=verbose,
            name='MLPP'
        )

    def __create_model(self):
        model = Sequential()
        model.add(Dense(self.hidden_size))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(self.dropout_rate))

        if self.no_of_classes > 2:
            model.add(Dense(self.no_of_classes,
                            activation='softmax'))

            model.compile(loss="categorical_crossentropy",
                          optimizer=Adam(self.learning_rate),
                          metrics=["accuracy"])

        else:
            model.add(Dense(1,
                            activation="sigmoid"))

            model.compile(loss="binary_crossentropy",
                          optimizer=Adam(self.learning_rate),
                          metrics=["accuracy"])

        return model
