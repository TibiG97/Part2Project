from utils import get_directory
from Data_Processing.synthetic_data_loader import SyntheticDataLoader
from Evaluation_Module.nest_cross_val import nested_cross_validation
from ML_Module.baseline_models import RandomForest
from ML_Module.baseline_models import KNeighbours
from ML_Module.baseline_models import LogRegression
from Data_Processing.data_creation import create_dataset_1

import sys


# mutag_dataset = load_local_data(get_directory(), 'node2')
def do_stuff(X, y):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from utils import merge_splits
    import numpy as np
    classifiers = [RandomForest(depth=5, estimators=10, features=1), KNeighbours(neighbours=3, p_dist=2),
                   LogRegression()]

    print(X)
    print(len(X))
    print(X[0])
    print(len(X[0]))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    for classifier in classifiers:
        classifier.train(X_train, y_train)
        predictions = classifier.predict_class(X_test)
        print(accuracy_score(y_test, predictions), file=open('results.txt', 'w'))


def main():
    create_dataset_1('NEWSET',
                     [1, 2, 3, 4, 5, 6, 7, 8, 9],
                     9,
                     [100] * 9,
                     [[0.0, 0.7, 0.2, 0.1, 0.0, 0.0], [0.7, 0.2, 0.0, 0.0, 0.1, 0.0], [0.0, 0.0, 0.7, 0.1, 0.0, 0.2],
                      [0.1, 0.1, 0.1, 0.1, 0.3, 0.3], [0.2, 0.2, 0.6, 0.0, 0.0, 0.0], [0.2, 0.2, 0.2, 0.2, 0.2, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.6, 0.4], [0.0, 0.0, 0.9, 0.1, 0.0, 0.0], [0.1, 0.8, 0.1, 0.0, 0.0, 0.0]],
                     [[0.7, 0.3, 0.0, 0.0, 0.0], [0.0, 0.0, 0.7, 0.3, 0.0], [0.0, 0.7, 0.0, 0.0, 0.3],
                      [0.2, 0.2, 0.2, 0.2, 0.2], [0.5, 0.5, 0.0, 0.0, 0.0], [0.7, 0.3, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.7, 0.3, 0.0], [0.0, 0.7, 0.0, 0.0, 0.3], [0.2, 0.2, 0.2, 0.2, 0.2]],
                     [[0.7, 0.3, 0.0, 0.0], [0.0, 0.8, 0.2, 0.0], [0.2, 0.0, 0.8, 0.0], [0.3, 0.3, 0.4, 0.0],
                      [0.3, 0.7, 0.0, 0.0], [0.7, 0.3, 0.0, 0.0], [0.0, 0.8, 0.2, 0.0], [0.2, 0.0, 0.8, 0.0],
                      [0.3, 0.3, 0.4, 0.0]])

    patchy_data_loader = SyntheticDataLoader('NEWSET', 'patchy_san')
    baselines_data_loader = SyntheticDataLoader('NEWSET', 'baselines')
    mutag_dataset = patchy_data_loader.load_synthetic_data_set()
    X, y, no_of_classes = mutag_dataset
    Xx, yy, C = baselines_data_loader.load_synthetic_data_set()

    # do_stuff(Xx, yy)

    '''
    old_dataset = load_local_data(get_directory(), 'node2')
    xa, ya = zip(*old_dataset)
    
    print(X[0].nodes())
    print(X[0].values())
    print(X[0].edges())
    
    print(xa[0].nodes())
    print(xa[0].edges())
    print(xa[0].values())
    '''

    nested_cross_validation(X, y, no_of_classes, 10, 1)
    # do_stuff(Xx, yy)


if __name__ == "__main__":
    main()
