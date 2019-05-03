from Data_Processing.data_loader import DataLoader
from Data_Processing.synthetic_data_creator import SyntheticDataGenerator

from constants import LOG_DIRS
from constants import LOG_REG_GRID
from constants import KNN_GRID
from constants import RF_GRID

from ML_Module.mlp import MultilayerPerceptron
from ML_Module.cnn import ConvolutionalNeuralNetwork
from ML_Module.baseline_models import LogRegression
from ML_Module.baseline_models import KNeighbours
from ML_Module.baseline_models import RandomForest
from Evaluation_Module.nest_cross_val import nested_cross_validation

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignores tensorflow warning for not supported CPU features


def demo():
    SyntheticDataGenerator.create_dataset(name='TEST',
                                          depth=3,
                                          no_of_classes=6,
                                          no_of_graphs_per_class=[500, 500, 500, 500, 500, 500],
                                          cmd_line_dist=[
                                              [0.1, 0.08, 0.08, 0.08, 0.08, 0.1, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08],
                                              [0.08, 0.1, 0.08, 0.08, 0.08, 0.08, 0.1, 0.08, 0.08, 0.08, 0.08, 0.08],
                                              [0.08, 0.08, 0.1, 0.08, 0.08, 0.08, 0.08, 0.1, 0.08, 0.08, 0.08, 0.08],
                                              [0.08, 0.08, 0.08, 0.1, 0.08, 0.08, 0.08, 0.08, 0.1, 0.08, 0.08, 0.08],
                                              [0.08, 0.08, 0.08, 0.08, 0.1, 0.08, 0.08, 0.08, 0.08, 0.1, 0.08, 0.08],
                                              [0.08, 0.08, 0.08, 0.08, 0.08, 0.1, 0.08, 0.08, 0.08, 0.08, 0.1, 0.08]],
                                          login_name_dist=[[0.25, 0.15, 0.15, 0.15, 0.15, 0.15],
                                                           [0.15, 0.25, 0.15, 0.15, 0.15, 0.15],
                                                           [0.15, 0.15, 0.25, 0.15, 0.15, 0.15],
                                                           [0.15, 0.15, 0.15, 0.25, 0.15, 0.15],
                                                           [0.15, 0.15, 0.15, 0.15, 0.25, 0.15],
                                                           [0.15, 0.15, 0.15, 0.15, 0.15, 0.25]],
                                          euid_dist=[[0.1, 0.2, 0.2, 0.3, 0.2],
                                                     [0.2, 0.1, 0.3, 0.2, 0.2],
                                                     [0.2, 0.3, 0.1, 0.2, 0.2],
                                                     [0.3, 0.2, 0.2, 0.1, 0.2],
                                                     [0.2, 0.3, 0.2, 0.2, 0.1],
                                                     [0.2, 0.2, 0.3, 0.1, 0.2]],
                                          binary_file_dist=[[0.20, 0.25, 0.25, 0.30],
                                                            [0.25, 0.20, 0.30, 0.25],
                                                            [0.25, 0.30, 0.20, 0.25],
                                                            [0.30, 0.25, 0.25, 0.20],
                                                            [0.25, 0.30, 0.20, 0.25],
                                                            [0.25, 0.20, 0.30, 0.25]],
                                          history_len=4,
                                          degree_dist={'values': [1, 2, 3, 4],
                                                       'probs': [0.7, 0.2, 0.05, 0.05]},
                                          node_type_dist=[0.9, 0.1])

    graph_dataset, graph_labels, no_of_classes = DataLoader.load_synthetic_data_set(name='TEST',
                                                                                    target_model='patchy_san')
    attr_dataset, attr_labels, no_of_classes = DataLoader.load_synthetic_data_set(name='TEST',
                                                                                  target_model='baselines')
    X, y = DataLoader.load_log_files(LOG_DIRS)

    cnn = ConvolutionalNeuralNetwork(width=15,
                                     stride=1,
                                     rf_size=5,
                                     epochs=100,
                                     batch_size=64,
                                     learning_rate=0.001,
                                     dropout_rate=0.5,
                                     no_of_classes=no_of_classes)

    lrg = LogRegression(c=LOG_REG_GRID['c'][2],
                        penalty=LOG_REG_GRID['penalty'][1],
                        width=8,
                        stride=1,
                        rf_size=5)

    knn = KNeighbours(neighbours=30,
                      p_dist=1)

    rf = RandomForest(depth=5,
                      estimators=30,
                      samples_leaf=2,
                      samples_split=2)

    mlp = MultilayerPerceptron(batch_size=32,
                               epochs=100,
                               learning_rate=0.001,
                               dropout_rate=0.5,
                               init_mode='he',
                               hidden_size=256,
                               no_of_classes=no_of_classes,
                               verbose=2)

    X_train, X_test, y_train, y_test = train_test_split(graph_dataset, graph_labels, test_size=0.1, random_state=42)
    cnn.train(X_train, y_train)
    lrg.train(X_train, y_train)

    predictions = cnn.predict_class(X_test)
    with open('Results/CNN', 'a') as results:
        results.truncate(0)
        print(accuracy_score(y_test, predictions), file=results)
        print(confusion_matrix(y_test, predictions), file=results)

    predictions = lrg.predict_class(X_test)
    with open('Results/LRG', 'a') as results:
        results.truncate(0)
        print(accuracy_score(y_test, predictions), file=results)
        print(confusion_matrix(y_test, predictions), file=results)

    X_train, X_test, y_train, y_test = train_test_split(attr_dataset, attr_labels, test_size=0.1, random_state=42)
    knn.train(X_train, y_train)
    rf.train(X_train, y_train)

    predictions = knn.predict_class(X_test)
    with open('Results/KNN', 'a') as results:
        results.truncate(0)
        print(accuracy_score(y_test, predictions), file=results)
        print(confusion_matrix(y_test, predictions), file=results)

    predictions = rf.predict_class(X_test)
    with open('Results/RF', 'a') as results:
        results.truncate(0)
        print(accuracy_score(y_test, predictions), file=results)
        print(confusion_matrix(y_test, predictions), file=results)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    mlp.train(X_train, y_train)
    predictions = mlp.predict_class(X_test)
    with open('Results/MLP', 'a') as results:
        results.truncate(0)
        print(accuracy_score(y_test, predictions), file=results)
        print(confusion_matrix(y_test, predictions), file=results)


def main():
    demo()


if __name__ == "__main__":
    main()
