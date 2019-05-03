from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from ML_Module.sklearn_classifier import SKLearnModel
from ML_Module.patchy_san import ReceptiveFieldMaker

from constants import DUMMY

import numpy as np


class RandomForest(SKLearnModel):
    """
    Class representing RandomForest classifiers

    """

    def __init__(self,
                 depth: int,
                 estimators: int,
                 samples_split: int,
                 samples_leaf: int,
                 **kwargs):
        """
        Object Constructor

        :param depth: the maximum depth of the tree
        :param estimators: the number of trees in the forest
        :param samples_split: the minimum number of samples required to split an internal node
        :param samples_leaf: the minimum number of samples required to be at a leaf node
        """

        self.depth = depth
        self.estimators = estimators
        self.samples_split = samples_split
        self.samples_leaf = samples_leaf

        super(RandomForest, self).__init__(
            classifier=RandomForestClassifier(
                max_depth=depth,
                n_estimators=estimators,
                min_samples_split=samples_split,
                min_samples_leaf=samples_leaf
            ),
            process_data=None,
            name='RF'
        )


class KNeighbours(SKLearnModel):
    """
    Class representing KNeighbours classifiers

    """

    def __init__(self,
                 neighbours: int,
                 p_dist: int,
                 **kwargs):
        """
        Object Constructor

        :param neighbours: number of neighbors to use by default for kneighbors queries.
        :param p_dist: power parameter for the Minkowski metric
        """

        self.neighbours = neighbours
        self.p_dist = p_dist

        super(KNeighbours, self).__init__(
            classifier=KNeighborsClassifier(
                n_neighbors=neighbours,
                p=p_dist
            ),
            process_data=None,
            name='KNN'
        )


class LogRegression(SKLearnModel):
    """
    Class representing LogisticRegression classifiers

    """

    def __init__(self,
                 c: float,
                 penalty: str,
                 width: int,
                 stride: int,
                 rf_size: int,
                 dummy_value=DUMMY,
                 **kwargs):
        """
        Object Constructor

        :param c: inverse of regularization strength (must be a positive float)
        :param penalty: used to specify the norm used in the penalization
        """

        self.C = c
        self.penalty = penalty
        self.width = width
        self.stride = stride
        self.rf_size = rf_size
        self.dummy_value = dummy_value

        super(LogRegression, self).__init__(
            LogisticRegression(
                C=c,
                penalty=penalty,
                solver='liblinear',
                multi_class='auto',
                max_iter=250
            ),
            process_data=self.__process_data,
            name='LRG'
        )

    def __process_data(self,
                       data_set: np.array):
        """
        Private method that builds the receptive fields from raw data

        :param data_set: list of graph objects
        :return: input data for the CNN
        """

        n = len(data_set)
        train_logistic = list()

        for i in range(n):
            receptive_field = ReceptiveFieldMaker(data_set[i].nx_graph,
                                                  w=self.width,
                                                  k=self.rf_size,
                                                  s=self.stride,
                                                  dummy_value=self.dummy_value)
            for_log = receptive_field.make_()
            train_logistic.append(np.array(for_log).flatten())

        logistig_train_set = np.array(train_logistic)

        return logistig_train_set
