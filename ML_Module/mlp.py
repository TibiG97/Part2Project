from ML_Module.neural_net import NeuralNetwork

import os
import glob
import shutil
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelBinarizer

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
                 hidden_size: int,
                 input_size: int):

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.model = KerasClassifier(build_fn=self.__create_model,
                                     epochs=epochs,
                                     batch_size=batch_size,
                                     verbose=verbose)

        super(MultilayerPerceptron, self).__init__(
            classifier=self.classifier,
            process_data=self.process_data,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            init_mode=init_mode,
            no_of_classes=no_of_classes,
            verbose=verbose,
            name='MLP'
        )

    def __create_model(self):
        model = Sequential()
        model.add(Dense(self.hidden_size, input_shape=(self.input_size,)))
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

    def __process_data(self):
        return

    @staticmethod
    def copy_data(self, src_file_path, dst_file_path):
        if not os.path.exists(dst_file_path):
            os.mkdir(dst_file_path)
        for logfile in glob.glob(src_file_path + "/*.log"):
            if os.stat(logfile)[6] > 10000:
                logfile_name = logfile.split('/')[-1]
                shutil.copyfile(logfile, dst_file_path + "/" + logfile_name)

    @staticmethod
    def read_data(self, logfile_path):
        log_collection = pd.DataFrame()
        logs = pd.DataFrame()
        logfiles = glob.glob(logfile_path + "/*.log")  # Get list of log files
        for logfile in logfiles:
            logs = pd.read_csv(logfile, sep="\n", header=None, names=['data'])
            logs['type'] = logfile.split('/')[-1]
            # Add log file data and type to log collection
            log_collection = log_collection.append(logs)

        # Remove empty lines
        log_collection = log_collection.dropna()
        # Reset the index
        log_collection = log_collection.reset_index(drop=True)

        return log_collection
