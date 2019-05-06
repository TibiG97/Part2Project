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

from utils import randomise_order4
from utils import create_ensamble

import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignores tensorflow warning for not supported CPU features


def demo():
    """
    SyntheticDataGenerator.create_dataset(name='TEST',
                                          depth=3,
                                          no_of_classes=6,
                                          no_of_graphs_per_class=[500, 500, 500, 500, 500, 500],
                                          cmd_line_dist=SyntheticDataGenerator.generate_dist_within_l2_dist(6, 12,
                                                                                                            (0.0,
                                                                                                             0.2)),

                                          login_name_dist=SyntheticDataGenerator.generate_dist_within_l2_dist(6, 6,
                                                                                                              (0.0,
                                                                                                               0.2)),
                                          euid_dist=SyntheticDataGenerator.generate_dist_within_l2_dist(6, 5,
                                                                                                        (0.0,
                                                                                                         0.2)),

                                          binary_file_dist=SyntheticDataGenerator.generate_dist_within_l2_dist(6, 4,
                                                                                                               (0.0,
                                                                                                                0.2)),

                                          history_len=4,
                                          degree_dist={'values': [1, 2, 3, 4],
                                                       'probs': [0.5, 0.3, 0.1, 0.1]},
                                          node_type_dist=[0.8, 0.2])
    """

    graph_dataset, graph_labels, no_of_classes = DataLoader.load_synthetic_data_set(name='D3_H4_L02',
                                                                                    target_model='patchy_san')

    attr_dataset, attr_labels, no_of_classes = DataLoader.load_synthetic_data_set(name='D3_H4_L02',
                                                                                  target_model='baselines')

    X, y = DataLoader.load_log_files(LOG_DIRS)

    cnn = ConvolutionalNeuralNetwork(width=15,
                                     stride=1,
                                     rf_size=5,
                                     epochs=1000,
                                     batch_size=32,
                                     learning_rate=0.001,
                                     dropout_rate=0.3,
                                     verbose=2,
                                     init_mode='he_normal',
                                     no_of_classes=no_of_classes)

    lrg = LogRegression(c=1e5,
                        penalty='l2',
                        no_of_classes=no_of_classes)

    knn = KNeighbours(neighbours=500,
                      p_dist=2,
                      no_of_classes=no_of_classes)

    rf = RandomForest(depth=70,
                      estimators=400,
                      samples_leaf=4,
                      samples_split=10,
                      no_of_classes=no_of_classes)

    mlp = MultilayerPerceptron(batch_size=32,
                               epochs=100,
                               learning_rate=0.0005,
                               dropout_rate=0.4,
                               init_mode='glorot_uniform',
                               hidden_size=128,
                               no_of_classes=6,
                               verbose=2)

    """
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.1,
                                                        random_state=42,
                                                        shuffle=True)
    mlp.train(X_train, y_train)

    predictions_cnn = mlp.predict_class(X_test)

    with open('Results/temp', 'a') as results:
        results.truncate(0)
        print(accuracy_score(y_test, predictions_cnn), file=results)
        print(confusion_matrix(y_test, predictions_cnn), file=results)
    """


def main():
    demo()


if __name__ == "__main__":
    main()
