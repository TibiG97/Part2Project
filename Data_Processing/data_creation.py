import numpy as np
from random import randint
from utils import get_directory, create_directory, clear_directory
from constants import *


def create_file_vector(binary_file_dist: np.array):
    binary_file = [0] * BINARY_FILE_SIZE
    position = np.random.choice(BINARY_FILE_CHOICES, p=binary_file_dist)
    binary_file[position] = 1

    file_encoding = FILE_ENCODING + EMPTY_CMD_LINE + EMPTY_LOGIN_NAME + EMPTY_EUID + binary_file
    return file_encoding


def create_process_vector(cmd_line_dist: np.array,
                          login_name_dist: np.array,
                          euid_dist: np.array):
    cmd_line = [0] * CMD_LINE_SIZE
    position = np.random.choice(CMD_LINE_CHOICES, p=cmd_line_dist)
    cmd_line[position] = 1

    login_name = [0] * LOGIN_NAME_SIZE
    position = np.random.choice(LOGIN_NAME_CHOICES, p=login_name_dist)
    login_name[position] = 1

    euid = [0] * EUID_SIZE
    position = np.random.choice(EUID_CHOICES, p=euid_dist)
    euid[position] = 1

    process_encoding = PROCESS_ENCODING + cmd_line + login_name + euid + EMPTY_BINARY_FILE
    return process_encoding


def create_socket_vector():
    socket_encoding = SOCKET_ECODING + EMPTY_CMD_LINE + EMPTY_LOGIN_NAME + EMPTY_EUID + EMPTY_BINARY_FILE
    return socket_encoding


def create_dataset_2(name: str,
                     depth: int,
                     no_of_classes: int,
                     no_of_graphs_per_class: np.array,
                     cmd_line_dist: np.array,
                     login_name_dist: np.array,
                     euid_dist: np.array,
                     binary_file_dist: np.array,
                     history_len: int,
                     degree_dist: np.array,
                     node_type_dist: np.array):
    main_dir_path = create_directory(get_directory() + '/Data_Sets/Provenance_Graphs', name)

    for class_number in range(1, no_of_classes + 1):
        class_dir_path = create_directory(main_dir_path, 'Class_' + str(class_number))
        clear_directory(class_dir_path)

        property_file = open(main_dir_path + '/property_file', 'a')
        property_file.truncate(0)
        print(no_of_classes, file=property_file)

        for dir_size in no_of_graphs_per_class:
            print(dir_size, file=property_file, end=' ')

        for iterator in range(1, no_of_graphs_per_class[class_number - 1] + 1):
            graph_file = open(class_dir_path + '/provenance_graph_' + str(iterator), 'a')
            graph_file.truncate(0)

            node_counter = 0
            edges = list()
            files = list()

            for file in range(0, history_len):
                node_counter += 1

                files.append(node_counter)

                nodes_at_depth = dict()
                nodes_at_depth['0'] = [node_counter]

                for level in range(0, depth):
                    nodes = nodes_at_depth[str(level)]
                    nodes_at_depth[str(level + 1)] = list()
                    for node in nodes:
                        no_of_neighbours = np.random.choice(degree_dist['values'], p=degree_dist['probs'])

                        for neigh in range(0, no_of_neighbours):
                            node_counter += 1
                            edges.append((node, node_counter))
                            nodes_at_depth[str(level + 1)].append(node_counter)

            no_of_nodes = node_counter
            no_of_edges = node_counter - 1

            print(no_of_nodes, no_of_edges, file=graph_file)

            for index in range(1, len(files)):
                edges.append((files[index - 1], files[index]))

            for node in range(1, no_of_nodes + 1):
                if node in files:
                    attribute = create_file_vector(binary_file_dist[class_number - 1])
                    for value in attribute:
                        print(value, file=graph_file, end=' ')
                    print(file=graph_file)
                else:
                    attribute = create_process_vector(cmd_line_dist[class_number - 1],
                                                      login_name_dist[class_number - 1],
                                                      euid_dist[class_number - 1])
                    for value in attribute:
                        print(value, file=graph_file, end=' ')
                    print(file=graph_file)

            for edge in edges:
                print(edge[0], edge[1], file=graph_file)


def create_dataset_1(name: str,
                     process_types: np.array,
                     no_of_classes: int,
                     no_of_graphs_per_class: np.array,
                     login_name_dist: np.array,
                     euid_dist: np.array,
                     binary_file_dist: np.array):
    """
    Method that creates a basic 2-Node synthetic dataset

    :param name: name of the dataset
    :param process_types: list containing types of processes for each class
    :param no_of_classes: number of classes of graphs
    :param no_of_graphs_per_class: number of graphs in each class
    :param login_name_dist: prob dist of login names (for processes)
    :param euid_dist: probability distribution of euids (for processes)
    :param binary_file_dist: probability distribution of provenance binary file (for files)
    :return: a synthetic dataset that respects the given dist
    """

    main_dir_path = create_directory(get_directory() + '/Data_Sets/Provenance_Graphs', name)

    for class_number in range(1, no_of_classes + 1):

        class_dir_path = create_directory(main_dir_path, 'Class_' + str(class_number))
        clear_directory(class_dir_path)

        property_file = open(main_dir_path + '/property_file', 'a')
        property_file.truncate(0)
        print(no_of_classes, file=property_file)

        for dir_size in no_of_graphs_per_class:
            print(dir_size, file=property_file, end=' ')

        for iterator in range(1, no_of_graphs_per_class[class_number - 1] + 1):
            graph_file = open(class_dir_path + '/provenance_graph_' + str(iterator), 'a')
            graph_file.truncate(0)

            print(2, 1, file=graph_file)

            login_name = [0] * LOGIN_NAME_SIZE
            position = np.random.choice(LOGIN_NAME_CHOICES, p=login_name_dist[class_number - 1])
            login_name[position] = 1

            euid = [0] * EUID_SIZE
            position = np.random.choice(EUID_CHOICES, p=euid_dist[class_number - 1])
            euid[position] = 1

            binary_file = [0] * BINARY_FILE_SIZE
            position = np.random.choice(BINARY_FILE_CHOICES, p=binary_file_dist[class_number - 1])
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
