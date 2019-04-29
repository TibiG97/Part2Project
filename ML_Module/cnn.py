from keras.wrappers.scikit_learn import KerasClassifier

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization

from ML_Module.classifier import Classifier
from ML_Module.patchy_san import ReceptiveFieldMaker

import numpy as np
import time


class ConvolutionalNeuralNetwork(Classifier):
    """
    Class representing Convolutional Neural Networks

    """

    def __init__(self,
                 w,
                 s=1,
                 k=10,
                 labeling_procedure_name='betweeness',
                 epochs=150, batch_size=25,
                 verbose=0,
                 use_node_deg=False,
                 use_preprocess_data=False,
                 gpu=False,
                 multiclass=None,
                 one_hot=0,
                 attr_dim=1,
                 dummy_value=-1,
                 parameters=[]):
        """

        :param w:
        :param s: length of the stride
        :param k: receptive field size
        :param labeling_procedure_name:
        :param epochs:
        :param batch_size:
        :param verbose:
        :param use_node_deg:
        :param use_preprocess_data:
        :param gpu:
        :param multiclass:
        :param one_hot:
        :param attr_dim:
        :param dummy_value:
        """

        """
        w : width parameter
        s: stride parameter
        k: receptive field size paremeter
        labeling_procedure_name : the labeling procedure for ranking the nodes between them. Only betweeness centrality is implemented.
        epochs: number of epochs for the CNN
        batch_size : batch size for training the CNN
        use_node_deg : wether to use node degree as label for unlabeled graphs (IMDB for e.g)
        multiclass : if the classification is not binary it is the number of classes
        one_hot : if nodes attributes are discrete it is the number of unique attributes
        attr_dim : if nodes attributes are multidimensionnal it is the dimension of the attributes
        dummy_value  which value should be used for dummy nodes (see paper)
        """
        self.w = w
        self.s = s
        self.k = k
        self.labeling_procedure_name = labeling_procedure_name
        self.epochs = epochs
        self.use_node_deg = use_node_deg
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_preprocess_data = use_preprocess_data
        self.gpu = gpu
        self.multiclass = multiclass
        self.one_hot = one_hot
        self.attr_dim = attr_dim
        self.dummy_value = dummy_value
        self.model = KerasClassifier(build_fn=self.__create_model
                                     , epochs=self.epochs,
                                     batch_size=self.batch_size, verbose=self.verbose)
        self.times_process_details = {}
        self.times_process_details['normalized_subgraph'] = []
        self.times_process_details['neigh_assembly'] = []
        self.times_process_details['canonicalizes'] = []
        self.times_process_details['compute_subgraph_ranking'] = []
        self.times_process_details['labeling_procedure'] = []
        self.times_process_details['first_labeling_procedure'] = []

        if self.one_hot > 0:
            self.attr_dim = self.one_hot

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
        if self.multiclass is not None:
            model.add(Dense(self.multiclass, activation='softmax'))
            model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        else:
            model.add(Dense(1, activation="sigmoid"))
            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

    def __process_data(self, X, y=None):  # X is a list of Graph objects
        """
        Private method that builds the receptive fields from raw data

        :param X: list of graph objects
        :param y: list of class labels for each graph
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
        if y is not None:
            y_preprocessed = [y[i] for i in range(n)]
            return X_preprocessed, y_preprocessed
        else:
            return X_preprocessed

    def train(self, X, y=None):
        """
        :param X: list of graph objects
        :param y: list of class labels for each graph
        """

        if not self.use_preprocess_data:
            X_preprocessed, y_preprocessed = self.__process_data(X, y)
        else:
            X_preprocessed = X
            y_preprocessed = y

        start = time.time()

        history = self.model.fit(X_preprocessed, y_preprocessed)
        # plot_accuracy_vs_epoch(history)
        print(history.history.keys())
        end = time.time()
        print('Time fit data in s', end - start)

    def predict_class(self, X):
        """
        :param X: list of graphs for which to make predictions
        :return: list of predicted classes
        """

        if not self.use_preprocess_data:
            X_preprocessed = self.__process_data(X)
        else:
            X_preprocessed = X
        y_pred_keras = self.model.predict(X_preprocessed)

        return y_pred_keras

    def predict_probs(self,
                      test_set):
        """
        :param test_set: list of graphs for which to make predictions
        :return: CNN's probability predictions for each class
        """

        return self.model.predict_proba(test_set)

    def save_model(self,
                   model_path: str):
        """
        :param model_path: path where to save the model
        """

        self.model.model.save(model_path)

    def load_model(self,
                   model_path: str):
        """
        :param model_path: path from where to load the model
        """

        self.model = KerasClassifier(
            build_fn=load_model(model_path).compile(loss="categorical_crossentropy", optimizer="adam",
                                                    metrics=["accuracy"]))
