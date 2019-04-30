from Data_Processing.synthetic_data_loader import SyntheticDataLoader

from Evaluation_Module.nest_cross_val import nested_cross_validation

from Data_Processing.data_creation import create_dataset_1


def main():
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

    patchy_data_loader = SyntheticDataLoader('NEWSET', 'patchy_san')
    graph_dataset, graph_labels, no_of_classes = patchy_data_loader.load_synthetic_data_set()

    baselines_data_loader = SyntheticDataLoader('NEWSET', 'baselines')
    attr_dataset, attr_labels, no_of_classes = baselines_data_loader.load_synthetic_data_set()

    nested_cross_validation(data_set=attr_dataset,
                            labels=attr_labels,
                            model_name='LRG',
                            tune_parameters=True,
                            no_of_classes=no_of_classes,
                            no_of_outer_folds=10,
                            no_of_inner_folds=10,
                            no_of_samples=30)


if __name__ == "__main__":
    main()
