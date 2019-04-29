from ML_Module.neural_net import NeuralNetwork

import os
import glob
import shutil
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation, Dense, Dropout
from keras.optimizers import Adam

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class MultilayerPerceptron(NeuralNetwork):
    """
    Class representing Multilayer Peceptrons

    """

    def __init__(self,
                 hidden_size,
                 num_classes,
                 dropout,
                 input_size,
                 criterion,
                 optimiser,
                 batch_size,
                 num_epochs):
        self.name = 'MLP'
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.input_size = input_size
        self.criterion = criterion
        self.optimiser = optimiser
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.mlp = self.__create_model()
        self.history = None

    def __create_model(self):
        nn = Sequential()
        nn.add(Dense(self.hidden_size, input_shape=(self.input_size,)))
        nn.add(Activation('relu'))
        nn.add(Dropout(self.dropout))
        nn.add(Dense(self.num_classes))
        nn.add(Activation('softmax'))
        nn.summary()

        return nn

    def __process_data(self):
        return

    def copy_data(self, src_file_path, dst_file_path):
        if not os.path.exists(dst_file_path):
            os.mkdir(dst_file_path)
        for logfile in glob.glob(src_file_path + "/*.log"):
            if os.stat(logfile)[6] > 10000:
                logfile_name = logfile.split('/')[-1]
                shutil.copyfile(logfile, dst_file_path + "/" + logfile_name)

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

    def train(self,
              X_train,
              y_train):
        self.mlp.compile(loss=self.criterion,
                         optimizer=self.optimiser,
                         metrics=['accuracy'])

        self.history = self.mlp.fit(X_train, y_train,
                                    batch_size=self.batch_size,
                                    epochs=self.num_epochs,
                                    verbose=1,
                                    validation_split=0.1)
