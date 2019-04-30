from keras.wrappers.scikit_learn import KerasClassifier

from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam

from ML_Module.neural_net import NeuralNetwork
from ML_Module.patchy_san import ReceptiveFieldMaker

from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, make_scorer

import numpy as np
import time

from ML_Module.classifier import Classifier


class ConvolutionalNeuralNetwork(NeuralNetwork):
    """
    Class representing Convolutional Neural Networks

    """

    def __init__(self,
                 w,
                 s=1,
                 k=10,
                 labeling_procedure_name='betweeness',
                 epochs=150,
                 batch_size=25,
                 verbose=0,
                 use_node_deg=False,
                 no_of_classes=None,
                 one_hot=0,
                 attr_dim=1,
                 dummy_value=-1,
                 parameters=[]):
        """

        :param w: width parameter
        :param s: length of the stride
        :param k: receptive field size
        :param labeling_procedure_name: the labeling procedure for ranking the nodes between them
        :param epochs: number of epochs for the CNN
        :param batch_size: batch size for training the CNN
        :param verbose:
        :param use_node_deg: wether to use node degree as label for unlabeled graphs
        :param no_of_classes: number of classes
        :param one_hot: if nodes attributes are discrete it is the number of unique attributes
        :param attr_dim: if nodes attributes are multidimensionnal it is the dimension of the attributes
        :param dummy_value:  which value should be used for dummy nodes
        """

        self.w = w
        self.s = s
        self.k = k
        self.labeling_procedure_name = labeling_procedure_name
        self.attr_dim = attr_dim
        self.use_node_deg = use_node_deg

        self.one_hot = one_hot
        self.dummy_value = dummy_value
        self.times_process_details = {}
        self.times_process_details['normalized_subgraph'] = []
        self.times_process_details['neigh_assembly'] = []
        self.times_process_details['canonicalizes'] = []
        self.times_process_details['compute_subgraph_ranking'] = []
        self.times_process_details['labeling_procedure'] = []
        self.times_process_details['first_labeling_procedure'] = []
        self.no_of_classes = no_of_classes

        if self.one_hot > 0:
            self.attr_dim = self.one_hot

        self.model = KerasClassifier(build_fn=self.__create_model,
                                     epochs=epochs,
                                     batch_size=batch_size,
                                     verbose=verbose)

        super(ConvolutionalNeuralNetwork, self).__init__(
            classifier=self.model,
            process_data=self.__process_data,
            epochs=epochs,
            no_of_classes=no_of_classes,
            verbose=verbose,
            batch_size=batch_size,
            name='CNN'
        )

    def __create_model(self):
        """
        Private method that builds the NN architecture of the model

        :return: the built NN model
        """

        model = Sequential()
        model.add(Conv1D(filters=16, kernel_size=self.k, strides=self.k, input_shape=(self.w * self.k, self.attr_dim)))
        model.add(Conv1D(filters=8, kernel_size=10, strides=1))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(128, activation="relu", name='embedding_layer'))
        model.add(Dropout(0.5))
        if self.no_of_classes > 2:
            model.add(Dense(self.no_of_classes, activation='softmax'))
            model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.005), metrics=["accuracy"])
        else:
            model.add(Dense(1, activation="sigmoid"))
            model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.005), metrics=["accuracy"])
        return model

    def __process_data(self, X):  # X is a list of Graph objects
        """
        Private method that builds the receptive fields from raw data

        :param X: list of graph objects
        :return: input data for the CNN
        """

        start = time.time()
        n = len(X)
        train = []
        for i in range(n):
            rfMaker = ReceptiveFieldMaker(X[i].nx_graph, w=self.w, k=self.k, s=self.s
                                          , labeling_procedure_name=self.labeling_procedure_name
                                          , use_node_deg=self.use_node_deg, one_hot=self.one_hot,
                                          dummy_value=self.dummy_value)
            forcnn = rfMaker.make_()
            self.times_process_details['neigh_assembly'].append(np.sum(rfMaker.all_times['neigh_assembly']))
            self.times_process_details['normalized_subgraph'].append(np.sum(rfMaker.all_times['normalized_subgraph']))
            self.times_process_details['canonicalizes'].append(np.sum(rfMaker.all_times['canonicalizes']))
            self.times_process_details['compute_subgraph_ranking'].append(
                np.sum(rfMaker.all_times['compute_subgraph_ranking']))
            self.times_process_details['labeling_procedure'].append(np.sum(rfMaker.all_times['labeling_procedure']))
            self.times_process_details['first_labeling_procedure'].append(
                np.sum(rfMaker.all_times['first_labeling_procedure']))

            # train.append(np.array(forcnn))
            train.append(np.array(forcnn).flatten().reshape(self.k * self.w, self.attr_dim))

        X_preprocessed = np.array(train)
        end = time.time()
        print('Time preprocess data in s', end - start)

        return X_preprocessed
