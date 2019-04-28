import numpy as np
from random import randint
from utils import get_parent_directory, create_directory
from constants import *


def create_data():
    graph_labels = open('graph_labels.txt', 'a')
    graph_labels.truncate(0)
    for interator in range(0, 100):
        print(1, file=graph_labels)
    for iterator in range(0, 100):
        print(-1, file=graph_labels)

    graph_indicator = open('graph_indicator.txt', 'a')
    graph_indicator.truncate(0)
    for iterator in range(1, 201):
        for i in range(0, 10):
            print(iterator, file=graph_indicator)

    graph_edges = open('graph_edges.txt', 'a')
    graph_edges.truncate(0)
    for iterator in range(0, 100):
        base = 10 * iterator
        print(str(base + 1) + ', ' + str(base + 4), file=graph_edges)
        print(str(base + 2) + ', ' + str(base + 4), file=graph_edges)
        print(str(base + 3) + ', ' + str(base + 4), file=graph_edges)
        print(str(base + 4) + ', ' + str(base + 5), file=graph_edges)
        print(str(base + 6) + ', ' + str(base + 5), file=graph_edges)
        print(str(base + 7) + ', ' + str(base + 5), file=graph_edges)
        print(str(base + 8) + ', ' + str(base + 5), file=graph_edges)
        print(str(base + 5) + ', ' + str(base + 10), file=graph_edges)
        print(str(base + 9) + ', ' + str(base + 10), file=graph_edges)
    for iterator in range(100, 200):
        base = 10 * iterator
        print(str(base + 1) + ', ' + str(base + 8), file=graph_edges)
        print(str(base + 2) + ', ' + str(base + 8), file=graph_edges)
        print(str(base + 3) + ', ' + str(base + 8), file=graph_edges)
        print(str(base + 4) + ', ' + str(base + 8), file=graph_edges)
        print(str(base + 5) + ', ' + str(base + 8), file=graph_edges)
        print(str(base + 6) + ', ' + str(base + 8), file=graph_edges)
        print(str(base + 7) + ', ' + str(base + 8), file=graph_edges)
        print(str(base + 8) + ', ' + str(base + 9), file=graph_edges)
        print(str(base + 9) + ', ' + str(base + 10), file=graph_edges)

    node_attributes = open('node_attributes.txt', 'a')
    node_attributes.truncate(0)
    for iterator in range(0, 100):
        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0',
              file=node_attributes)
        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0',
              file=node_attributes)
        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0',
              file=node_attributes)
        print('1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0',
              file=node_attributes)
        print('1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0',
              file=node_attributes)
        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0',
              file=node_attributes)
        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0',
              file=node_attributes)
        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0',
              file=node_attributes)
        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0',
              file=node_attributes)
        print('1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0',
              file=node_attributes)
    for iterator in range(100, 200):
        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0',
              file=node_attributes)
        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0',
              file=node_attributes)
        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0',
              file=node_attributes)
        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0',
              file=node_attributes)
        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0',
              file=node_attributes)
        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0',
              file=node_attributes)
        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0',
              file=node_attributes)
        print('1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0',
              file=node_attributes)
        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0',
              file=node_attributes)
        print('1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0',
              file=node_attributes)


def create_basic_ssh_vs_sleep():
    graph_labels = open('graph_labels.txt', 'a')
    graph_labels.truncate(0)
    for interator in range(0, 500):
        print(1, file=graph_labels)
    for iterator in range(0, 500):
        print(-1, file=graph_labels)

    graph_indicator = open('graph_indicator.txt', 'a')
    graph_indicator.truncate(0)
    for iterator in range(1, 1001):
        for i in range(0, 2):
            print(iterator, file=graph_indicator)

    graph_edges = open('graph_edges.txt', 'a')
    graph_edges.truncate(0)
    for iterator in range(1, 2000, 2):
        print(str(iterator) + ', ' + str(iterator + 1), file=graph_edges)

    node_attributes = open('node_attributes.txt', 'a')
    node_attributes.truncate(0)
    for iterator in range(0, 500):
        indicator = randint(1, 10)
        print('1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ', file=node_attributes,
              end='')
        if indicator <= 7:
            print('0, 0, 1, 0', file=node_attributes)
        else:
            print('0, 0, 0, 1', file=node_attributes)

        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, ', file=node_attributes, end='')
        if indicator <= 7:
            print('0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0', file=node_attributes)
        elif indicator <= 9:
            print('1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0', file=node_attributes)
        else:
            print('0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0', file=node_attributes)

    for iterator in range(500, 1000):
        indicator = randint(1, 10)
        print('1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ', file=node_attributes,
              end='')
        if indicator <= 8:
            print('0, 0, 0, 1', file=node_attributes)
        else:
            print('0, 1, 0, 0', file=node_attributes)

        print('0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ', file=node_attributes, end='')
        if indicator <= 7:
            print('0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0', file=node_attributes)
        elif indicator <= 9:
            print('1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0', file=node_attributes)
        else:
            print('0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0', file=node_attributes)


def create_basic_4_class_comparison():
    graph_labels = open('graph_labels.txt', 'a')
    graph_labels.truncate(0)
    for interator in range(0, 500):
        print(1, file=graph_labels)
    for iterator in range(0, 500):
        print(2, file=graph_labels)
    for iterator in range(0, 500):
        print(3, file=graph_labels)
    for iterator in range(0, 500):
        print(4, file=graph_labels)

    graph_indicator = open('graph_indicator.txt', 'a')
    graph_indicator.truncate(0)
    for iterator in range(1, 2001):
        for i in range(0, 2):
            print(iterator, file=graph_indicator)

    graph_edges = open('graph_edges.txt', 'a')
    graph_edges.truncate(0)
    for iterator in range(1, 4000, 2):
        print(str(iterator) + ', ' + str(iterator + 1), file=graph_edges)

    node_attributes = open('node_attributes.txt', 'a')
    node_attributes.truncate(0)
    for iterator in range(0, 500):
        indicator = randint(1, 10)
        print('1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ', file=node_attributes,
              end='')
        if indicator <= 7:
            print('0, 0, 1, 0', file=node_attributes)
        else:
            print('0, 0, 0, 1', file=node_attributes)

        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, ', file=node_attributes, end='')
        if indicator <= 7:
            print('0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0', file=node_attributes)
        elif indicator <= 9:
            print('1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0', file=node_attributes)
        else:
            print('0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0', file=node_attributes)

    for iterator in range(500, 1000):
        indicator = randint(1, 10)
        print('1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ', file=node_attributes,
              end='')
        if indicator <= 8:
            print('0, 0, 0, 1', file=node_attributes)
        else:
            print('0, 1, 0, 0', file=node_attributes)

        print('0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ', file=node_attributes, end='')
        if indicator <= 7:
            print('0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0', file=node_attributes)
        elif indicator <= 9:
            print('1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0', file=node_attributes)
        else:
            print('0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0', file=node_attributes)

    for iterator in range(500, 1000):
        indicator = randint(1, 10)
        print('1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ', file=node_attributes,
              end='')
        if indicator <= 8:
            print('1, 0, 0, 0', file=node_attributes)
        else:
            print('1, 0, 0, 0', file=node_attributes)

        print('0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, ', file=node_attributes, end='')
        if indicator <= 7:
            print('0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0', file=node_attributes)
        elif indicator <= 9:
            print('0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0', file=node_attributes)
        else:
            print('0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0', file=node_attributes)

    for iterator in range(500, 1000):
        indicator = randint(1, 10)
        print('1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ', file=node_attributes,
              end='')
        if indicator <= 8:
            print('0, 0, 0, 1', file=node_attributes)
        else:
            print('0, 1, 0, 0', file=node_attributes)

        print('0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, ', file=node_attributes, end='')
        if indicator <= 7:
            print('0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0', file=node_attributes)
        elif indicator <= 9:
            print('0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0', file=node_attributes)
        else:
            print('0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0', file=node_attributes)


def create_dataset_1(name: str,
                     process_types: list,
                     no_of_classes: int,
                     no_of_graphs: list,
                     login_name_distributions: list,
                     euid_distributions: list,
                     binary_file_distributions: list):
    """
    Method that creates a basic 2-Node synthetic dataset

    :param name: name of the dataset
    :param process_types: list containing types of processes for each class
    :param no_of_classes: number of classes of graphs
    :param no_of_graphs: number of graphs in each class
    :param login_name_distributions: prob dist of login names (for processes)
    :param euid_distributions: probability distribution of euids (for processes)
    :param binary_file_distributions: probability distribution of provenance binary file (for files)
    :return: a synthetic dataset that respects the given distributions
    """

    main_dir_path = create_directory(get_parent_directory() + '/DataSets', name)

    for class_number in range(1, no_of_classes + 1):

        class_dir_path = create_directory(main_dir_path, 'Class_' + str(class_number))
        property_file = open(main_dir_path + '/property_file', 'a')
        property_file.truncate(0)
        print(no_of_classes, file=property_file)

        for dir_size in no_of_graphs:
            print(dir_size, file=property_file, end=' ')

        for iterator in range(1, no_of_graphs[class_number - 1] + 1):
            graph_file = open(class_dir_path + '/provenance_graph_' + str(iterator), 'a')
            graph_file.truncate(0)

            print(2, 1, file=graph_file)

            login_name = [0] * LOGIN_NAME_SIZE
            position = np.random.choice(LOGIN_NAME_CHOICES, p=login_name_distributions[class_number - 1])
            login_name[position] = 1

            euid = [0] * EUID_SIZE
            position = np.random.choice(EUID_CHOICES, p=euid_distributions[class_number - 1])
            euid[position] = 1

            binary_file = [0] * BINARY_FILE_SIZE
            position = np.random.choice(BINARY_FILE_CHOICES, p=binary_file_distributions[class_number - 1])
            binary_file[position] = 1

            node1 = FILE_ENCODING + EMPTY_CMD_LINE + EMPTY_LOGIN_NAME + EMPTY_EUID + binary_file
            node2 = PROCESS_ENCODING + CMD_LINE[
                process_types[class_number - 1] - 1] + login_name + euid + EMPTY_BINARY_FILE

            for element in node1:
                print(element, file=graph_file, end=' ')
            print(file=graph_file)
            for element in node2:
                print(element, file=graph_file, end=' ')
            print(file=graph_file)
            print(1, 2, file=graph_file)


create_dataset_1('NEWSET',
                 [1, 2, 3],
                 3,
                 [500, 500, 500],
                 [[0.0, 0.7, 0.2, 0.1, 0.0, 0.0], [0.7, 0.2, 0.0, 0.0, 0.1, 0.0], [0.0, 0.0, 0.7, 0.1, 0.0, 0.2]],
                 [[0.7, 0.3, 0.0, 0.0, 0.0], [0.0, 0.0, 0.7, 0.3, 0.0], [0.0, 0.7, 0.0, 0.0, 0.3]],
                 [[0.7, 0.3, 0.0, 0.0], [0.0, 0.8, 0.2, 0.0], [0.2, 0.0, 0.8, 0.0]])

# create_basic_ssh_vs_sleep()
# create_basic_4_class_comparison()
