from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Model, Sequential
from keras.layers import Conv1D, Dense, Dropout, Flatten
from ML_Module.Patchy_San import ReceptiveFieldMaker
import numpy as np
import time
import tensorflow as tf


class PSCN:
    def __init__(self, w, s=1, k=10
                 , labeling_procedure_name='betweeness'
                 , epochs=150, batch_size=25
                 , verbose=0
                 , use_node_deg=False
                 , use_preprocess_data=False
                 , gpu=False
                 , multiclass=None
                 , one_hot=0
                 , attr_dim=1
                 , dummy_value=-1):
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
        self.model = KerasClassifier(build_fn=self.create_model
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

    def create_model(self):
        model = Sequential()
        model.add(Conv1D(filters=16, kernel_size=self.k, strides=self.k, input_shape=(self.w * self.k, self.attr_dim)))
        model.add(Conv1D(filters=8, kernel_size=10, strides=1))
        model.add(Flatten())
        model.add(Dense(128, activation="relu", name='embedding_layer'))
        model.add(Dropout(0.5))
        if self.multiclass is not None:
            model.add(Dense(self.multiclass, activation='softmax'))
            model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
        else:
            model.add(Dense(1, activation="sigmoid"))
            model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
        return model

    def process_data(self, X, y=None):  # X is a list of Graph objects
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

            train.append(np.array(forcnn).flatten().reshape(self.k * self.w, self.attr_dim))

        X_preprocessed = np.array(train)
        end = time.time()
        print('Time preprocess data in s', end - start)
        if y is not None:
            y_preprocessed = [y[i] for i in range(n)]
            return X_preprocessed, y_preprocessed
        else:
            return X_preprocessed

    def fit(self, X, y=None):
        if not self.use_preprocess_data:
            X_preprocessed, y_preprocessed = self.process_data(X, y)
        else:
            X_preprocessed = X
            y_preprocessed = y
        start = time.time()
        if self.gpu:
            with tf.device("/device:GPU:1"):
                if self.verbose > 0:
                    print('Go for GPU')
                self.model.fit(X_preprocessed, y_preprocessed)
        else:
            self.model.fit(X_preprocessed, y_preprocessed)
        end = time.time()
        print('Time fit data in s', end - start)

    def predict(self, X):
        if not self.use_preprocess_data:
            X_preprocessed = self.process_data(X)
        else:
            X_preprocessed = X
        return self.model.predict(X_preprocessed)

    def return_embedding(self, X):
        X_preprocessed = self.process_data(X)
        layer_output = Model(inputs=self.model.model.input,
                             outputs=self.model.model.get_layer('embedding_layer').output)
        return layer_output.predict(X_preprocessed)
