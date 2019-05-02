from Data_Processing.data_loader import DataLoader

from Evaluation_Module.nest_cross_val import nested_cross_validation, tune_mlp_parameters

from Data_Processing.synthetic_data_creator import SyntheticDataGenerator


def main():
    SyntheticDataGenerator.create_dataset_2(name='TEST',
                                            depth=4,
                                            no_of_classes=6,
                                            no_of_graphs_per_class=[150, 150, 150, 150, 150, 150],
                                            cmd_line_dist=[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                           [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                           [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                           [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                           [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                           [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                            0.0]],
                                            login_name_dist=[[0.2, 0.3, 0.0, 0.3, 0.2, 0.0],
                                                             [0.1, 0.2, 0.4, 0.2, 0.1, 0.0],
                                                             [0.2, 0.1, 0.5, 0.1, 0.1, 0.0],
                                                             [0.1, 0.3, 0.2, 0.1, 0.2, 0.1],
                                                             [0.2, 0.5, 0.0, 0.0, 0.0, 0.3],
                                                             [0.1, 0.1, 0.0, 0.4, 0.0, 0.4]],
                                            euid_dist=[[1, 0, 0, 0, 0],
                                                       [1, 0, 0, 0, 0],
                                                       [1, 0, 0, 0, 0],
                                                       [1, 0, 0, 0, 0],
                                                       [1, 0, 0, 0, 0],
                                                       [1, 0, 0, 0, 0]],
                                            binary_file_dist=[[1, 0, 0, 0],
                                                              [1, 0, 0, 0],
                                                              [1, 0, 0, 0],
                                                              [1, 0, 0, 0],
                                                              [1, 0, 0, 0],
                                                              [1, 0, 0, 0]],
                                            history_len=5,
                                            degree_dist={'values': [1, 2, 3, 4],
                                                         'probs': [0.7, 0.3, 0.0, 0.0]},
                                            node_type_dist=[])

    graph_dataset, graph_labels, no_of_classes = DataLoader.load_synthetic_data_set(name='TEST',
                                                                                    target_model='patchy_san')

    attr_dataset, attr_labels, no_of_classes = DataLoader.load_synthetic_data_set(name='TEST',
                                                                                  target_model='baselines')

    import numpy as np

    from constants import LOG_DIRS
    from utils import convert_labels_to_pos_neg

    X, y = DataLoader.load_log_files(LOG_DIRS)

    nested_cross_validation(data_set=graph_dataset,
                            labels=graph_labels,
                            model_name='CNN',
                            tune_parameters=False,
                            no_of_classes=no_of_classes,
                            no_of_outer_folds=10,
                            no_of_inner_folds=10,
                            no_of_samples=1000)

    from constants import LOG_REG_GRID, KNN_GRID, RF_GRID
    from ML_Module.baseline_models import LogRegression, KNeighbours, RandomForest
    import numpy as np

    model1 = LogRegression(c=LOG_REG_GRID['c'][5], penalty=LOG_REG_GRID['penalty'][1])
    model2 = KNeighbours(neighbours=20, p_dist=2)
    model3 = RandomForest(depth=5, estimators=15, samples_split=2, samples_leaf=2)
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    attr_labels = convert_labels_to_pos_neg(attr_labels)
    X_train, X_test, y_train, y_test = train_test_split(attr_dataset, attr_labels, test_size=0.1, random_state=42)

    model1.train(X_train, y_train)
    model2.train(X_train, y_train)
    model3.train(X_train, y_train)

    predictions1 = model1.predict_class(X_test)
    predictions2 = model2.predict_class(X_test)
    predictions3 = model3.predict_class(X_test)
    print('Logistic')
    print(accuracy_score(y_test, predictions1))
    print('KNN')
    print(accuracy_score(y_test, predictions2))
    print('RandomForest')
    print(accuracy_score(y_test, predictions3))


if __name__ == "__main__":
    main()
