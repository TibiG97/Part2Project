from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from ML_Module.sklearn_classifier import SKLearnModel


class RandomForest(SKLearnModel):

    def __init__(self,
                 depth: int,
                 estimators: int,
                 features: int):
        """
        Object Constructor

        :param depth:
        :param estimators:
        :param features:
        """
        self.depth = depth
        self.estimators = estimators
        self.features = features
        super(RandomForest, self).__init__(
            RandomForestClassifier(max_depth=depth, n_estimators=estimators, max_features=features))


class KNeighbours(SKLearnModel):

    def __init__(self,
                 neighbours: int):
        """
        Object Constructor

        :param neighbours:
        """
        self.neighbours = neighbours
        super(KNeighbours, self).__init__(
            KNeighborsClassifier(n_neighbors=neighbours))


class LogRegression(SKLearnModel):

    def __init__(self):
        """
        Object Constructor

        """
        super(LogRegression, self).__init__(LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000))
