from keras.wrappers.scikit_learn import KerasClassifier

from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import MaxPool1D
from keras.optimizers import Adam
from keras.regularizers import L1L2

from ML_Module.neural_net import NeuralNetwork
from ML_Module.patchy_san import ReceptiveFieldMaker

from constants import EMPTY_TIMES_DICT
from constants import DUMMY
from constants import ATTR_DIM

from copy import copy
import numpy as np


class ConvolutionalNeuralNetwork(NeuralNetwork):
    """
    Class representing Convolutional Neural Networks

    """

    def __init__(self,
                 width: int,
                 stride: int,
                 rf_size: int,
                 epochs: int,
                 batch_size: int,
                 learning_rate: float,
                 dropout_rate: float,
                 no_of_classes: int,
                 labeling_procedure_name='betweeness',
                 init_mode='he_normal',
                 verbose=2,
                 attr_dim=ATTR_DIM,
                 dummy_value=DUMMY):
        """
        :param width: width parameter
        :param stride: length of the stride
        :param rf_size: receptive field size
        :param labeling_procedure_name: the labeling procedure for ranking the nodes between them
        :param epochs: number of epochs for the CNN
        :param batch_size: batch size for training the CNN
        :param verbose: choose how to see training progress
        :param no_of_classes: number of classes
        :param attr_dim: if nodes attributes are multidimensionnal it is the dimension of the attributes
        :param dummy_value:  which value should be used for dummy nodes
        """

        self.width = width
        self.stride = stride
        self.rf_size = rf_size
        self.labeling_procedure_name = labeling_procedure_name

        self.attr_dim = attr_dim
        self.dummy_value = dummy_value

        self.times = copy(EMPTY_TIMES_DICT)

        self.model = KerasClassifier(build_fn=self.__create_model,
                                     epochs=epochs,
                                     batch_size=batch_size,
                                     verbose=verbose)

        super(ConvolutionalNeuralNetwork, self).__init__(
            classifier=self.model,
            process_data=self.__process_data,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            init_mode=init_mode,
            no_of_classes=no_of_classes,
            verbose=verbose,
            name='CNN'
        )

    def __create_model(self):
        """
        Private method that builds the NN architecture of the model

        :return: the compiled NN model
        """

        model = Sequential()

        model.add(Conv1D(filters=32,
                         kernel_size=self.rf_size,
                         strides=self.rf_size,
                         input_shape=(self.width * self.rf_size, self.attr_dim),
                         kernel_initializer=self.init_mode,
                         kernel_regularizer=L1L2(l1=0.0, l2=0.1)))

        model.add(Conv1D(filters=64,
                         kernel_size=self.rf_size,
                         strides=1))

        model.add(Flatten())

        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(self.dropout_rate))

        if self.no_of_classes > 2:
            model.add(Dense(self.no_of_classes))
            # model.add(BatchNormalization())
            model.add(Activation('softmax'))

            model.compile(loss="categorical_crossentropy",
                          optimizer=Adam(self.learning_rate),
                          metrics=["accuracy"])

        else:
            model.add(Dense(1))
            # model.add(BatchNormalization())
            model.add(Activation('sigmoid'))

            model.compile(loss="binary_crossentropy",
                          optimizer=Adam(self.learning_rate),
                          metrics=["accuracy"])

        return model

    def __process_data(self, data_set):
        """
        Private method that builds the receptive fields from raw data

        :param data_set: list of graph objects
        :return: input data for the CNN
        """

        n = len(data_set)
        train = list()

        for i in range(n):
            receptive_field = ReceptiveFieldMaker(data_set[i].nx_graph,
                                                  w=self.width,
                                                  k=self.rf_size,
                                                  s=self.stride,
                                                  labeling_procedure_name=self.labeling_procedure_name,
                                                  dummy_value=self.dummy_value)
            for_cnn = receptive_field.make_()
            train.append(np.array(for_cnn).flatten().reshape(self.rf_size * self.width, self.attr_dim))

        cnn_train_set = np.array(train)

        return cnn_train_set

    def get_rf_times(self,
                     receptive_field: ReceptiveFieldMaker):
        for entry in self.times.keys():
            self.times[entry].append(np.sum(receptive_field.all_times[entry]))
