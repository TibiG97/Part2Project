from Data_Processing.data_loader import DataLoader
from Data_Processing.synthetic_data_creator import SyntheticDataGenerator

from constants import LOG_DIRS

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

from rb_renaming import rename

import numpy as np
import os

from pickle import dump, load

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignores tensorflow warning for not supported CPU features


def synthetic_data_generator_demo():
    SyntheticDataGenerator.create_dataset(name='DEMO',
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


def one_fold_demo():
    X, y, names, paths = DataLoader.load_log_files(LOG_DIRS)

    graph_dataset, graph_labels, no_of_classes = DataLoader.load_synthetic_data_set(name='D3_H4_L02',
                                                                                    target_model='patchy_san')

    attr_dataset, attr_labels, no_of_classes = DataLoader.load_synthetic_data_set(name='D3_H4_L02',
                                                                                  target_model='baselines')

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

    x_train, x_test, y_train, y_test = train_test_split(graph_dataset, graph_labels, test_size=0.1, random_state=42)

    cnn.train(x_train, y_train)
    predictions = cnn.predict_class(x_test)
    print(accuracy_score(y_test, predictions))


def cross_valudation_demo():
    X, y, names, paths = DataLoader.load_log_files(LOG_DIRS)

    graph_dataset, graph_labels, no_of_classes = DataLoader.load_synthetic_data_set(name='D3_H4_L02',
                                                                                    target_model='patchy_san')

    attr_dataset, attr_labels, no_of_classes = DataLoader.load_synthetic_data_set(name='D3_H4_L02',
                                                                                  target_model='baselines')

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
                               epochs=1000,
                               learning_rate=0.0005,
                               dropout_rate=0.4,
                               init_mode='he_normal',
                               hidden_size=128,
                               no_of_classes=6,
                               verbose=2)

    knn.cross_validate(attr_dataset, attr_labels, 10, clear_file=True)
    rf.cross_validate(attr_dataset, attr_labels, 10, clear_file=True)
    mlp_average_acc, mlp_all_accuracies, mlp_all_predictions, mlp_model_history = mlp.cross_validate(X,
                                                                                                     y, 10,
                                                                                                     clear_file=True)

    cnn_average_acc, cnn_all_accuracies, cnn_all_predictions, cnn_model_history = cnn.cross_validate(graph_dataset,
                                                                                                     graph_labels, 10,
                                                                                                     clear_file=True)

    lrg_acc, lrg_all_acc, lrg_pred, h = create_ensamble(cnn_all_predictions, mlp_all_predictions, graph_labels, lrg, 10)


def ncv_demo():
    x, y, names, paths = DataLoader.load_log_files(LOG_DIRS)

    graph_dataset, graph_labels, no_of_classes = DataLoader.load_synthetic_data_set(name='D3_H4_L02',
                                                                                    target_model='patchy_san')

    attr_dataset, attr_labels, no_of_classes = DataLoader.load_synthetic_data_set(name='D3_H4_L02',
                                                                                  target_model='baselines')

    nested_cross_validation(data_set=graph_dataset,
                            labels=graph_labels,
                            model_name='CNN',
                            no_of_classes=no_of_classes,
                            no_of_outer_folds=2,
                            no_of_inner_folds=2,
                            no_of_samples=1)


def main():
    rename()
    return


if __name__ == "__main__":
    main()
