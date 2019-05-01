from Data_Processing.data_loader import DataLoader

from Evaluation_Module.nest_cross_val import nested_cross_validation, tune_mlp_parameters

from Data_Processing.data_creation import create_dataset_1


def main():
    '''
    create_dataset_1('NEWSET',
                     [1, 2, 3, 4, 5, 6, 7, 8, 9],
                     9,
                     [200] * 9,
                     [[0.0, 0.7, 0.2, 0.1, 0.0, 0.0], [0.7, 0.2, 0.0, 0.0, 0.1, 0.0], [0.0, 0.0, 0.7, 0.1, 0.0, 0.2],
                      [0.1, 0.1, 0.1, 0.1, 0.3, 0.3], [0.2, 0.2, 0.6, 0.0, 0.0, 0.0], [0.2, 0.2, 0.2, 0.2, 0.2, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.6, 0.4], [0.0, 0.0, 0.9, 0.1, 0.0, 0.0], [0.1, 0.8, 0.1, 0.0, 0.0, 0.0]],
                     [[0.7, 0.3, 0.0, 0.0, 0.0], [0.0, 0.0, 0.7, 0.3, 0.0], [0.0, 0.7, 0.0, 0.0, 0.3],
                      [0.2, 0.2, 0.2, 0.2, 0.2], [0.5, 0.5, 0.0, 0.0, 0.0], [0.7, 0.3, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.7, 0.3, 0.0], [0.0, 0.7, 0.0, 0.0, 0.3], [0.2, 0.2, 0.2, 0.2, 0.2]],
                     [[0.7, 0.3, 0.0, 0.0], [0.0, 0.8, 0.2, 0.0], [0.2, 0.0, 0.8, 0.0], [0.3, 0.3, 0.4, 0.0],
                      [0.3, 0.7, 0.0, 0.0], [0.7, 0.3, 0.0, 0.0], [0.0, 0.8, 0.2, 0.0], [0.2, 0.0, 0.8, 0.0],
                      [0.3, 0.3, 0.4, 0.0]])
    '''

    graph_dataset, graph_labels, no_of_classes = DataLoader.load_synthetic_data_set(name='NEWSET',
                                                                                    target_model='patchy_san')

    attr_dataset, attr_labels, no_of_classes = DataLoader.load_synthetic_data_set(name='NEWSET',
                                                                                  target_model='baselines')

    from utils import get_directory
    android_dir = get_directory() + '/DataSets/Logs/Android'
    apache_dir = get_directory() + '/DataSets/Logs/Apache'
    hadoop_dir = get_directory() + '/DataSets/Logs/Hadoop'
    open_dir = get_directory() + '/DataSets/Logs/OpenStack'
    spark_dir = get_directory() + '/DataSets/Logs/Spark'
    ssh_dir = get_directory() + '/DataSets/Logs/SSH'

    dirs = [android_dir,
            apache_dir,
            hadoop_dir,
            open_dir,
            spark_dir,
            ssh_dir]
    X, y = DataLoader.load_log_files(dirs)

    for iterator in range(0, 10):
        nested_cross_validation(data_set=X,
                                labels=y,
                                model_name='MLP',
                                tune_parameters=False,
                                no_of_classes=len(dirs),
                                no_of_outer_folds=10,
                                no_of_inner_folds=10,
                                no_of_samples=1000)

    from ML_Module.mlp import MultilayerPerceptron

    '''
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    from copy import copy
    from constants import MLP_GRID
    from random import randint
    from ML_Module.mlp import MultilayerPerceptron

    param_grid = copy(MLP_GRID)
    best_choices = None
    best_accuracy = 0

    for iterator in range(0, 50):

        choices = dict()
        for entry in param_grid.keys():
            position = randint(0, len(param_grid[entry]) - 1)
            choices[entry] = param_grid[entry][position]

        print(choices)
        model = MultilayerPerceptron(hidden_size=choices['hidden_size'],
                                     batch_size=choices['batch_size'],
                                     epochs=choices['epochs'],
                                     learning_rate=choices['learning_rate'],
                                     dropout_rate=choices['dropout_rate'],
                                     init_mode=choices['init_mode'],
                                     no_of_classes=6,
                                     verbose=0)
        acc = model.cross_validate(X, y, 10)

        print(acc)
        if acc > best_accuracy:
            best_accuracy = acc
            best_choices = choices

    print()
    print('FINAL')
    print(best_choices)
    print(best_accuracy)
    '''


if __name__ == "__main__":
    main()
