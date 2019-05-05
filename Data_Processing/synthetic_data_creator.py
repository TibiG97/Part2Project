import numpy as np
from random import randint

from utils import get_directory
from utils import create_directory
from utils import clear_directory

from math import sqrt

from constants import *


class SyntheticDataGenerator(object):

    @staticmethod
    def __l2_distance(prob_dist: list):
        """
        Private function that computed the l2 distance between discrete input dist and uniform dist

        :param prob_dist: input dist
        :return: l2 distance
        """

        dist = 0
        size = len(prob_dist)
        uniform = [1 / size] * size

        for iterator in range(0, size):
            dist += pow(prob_dist[iterator] - uniform[iterator], 2)
        dist = sqrt(dist)

        return dist

    @staticmethod
    def generate_dist_within_l2_dist(no_of_dist_required: int,
                                     size: int,
                                     interval: tuple):
        """
        Function that generates probability distributions which have l2 distance in given range

        :param no_of_dist_required: number of distributions to be generated
        :param size: size of the discrete dist
        :param interval: l2 distance intervals of the returned dist
        :return: distribution which respects the input parameters
        """
        all_dist = list()

        while True:
            dist = list()
            for it in range(0, size):
                dist.append(randint(0, 100))

            # normalise in (0,1)
            sum_dist = sum(dist)
            for it in range(0, size):
                dist[it] = dist[it] / sum_dist

            # check interval
            if interval[0] <= SyntheticDataGenerator.__l2_distance(dist) <= interval[1]:
                all_dist.append(dist)

            # check if we generated enough distributions
            if len(all_dist) == no_of_dist_required:
                return all_dist

    @staticmethod
    def __create_file_vector(binary_file_dist: np.array):
        """
        Function that creates a feature vector for a file node

        :param binary_file_dist: file type distribution
        :return: feature vector
        """

        binary_file = [0] * BINARY_FILE_SIZE
        position = np.random.choice(BINARY_FILE_CHOICES, p=binary_file_dist)
        binary_file[position] = 1

        file_encoding = FILE_ENCODING + EMPTY_CMD_LINE + EMPTY_LOGIN_NAME + EMPTY_EUID + binary_file
        return file_encoding

    @staticmethod
    def __create_process_vector(cmd_line_dist: np.array,
                                login_name_dist: np.array,
                                euid_dist: np.array):
        """
        Function that creates a feature vector for a process node

        :param cmd_line_dist: command line instructions distribution
        :param login_name_dist:  login name distribution
        :param euid_dist: euid values distribution
        :return: feature vector
        """
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

    @staticmethod
    def __create_socket_vector():
        """
        Function that created a feature vector for a socket node

        :return: feature vector
        """

        socket_encoding = SOCKET_ECODING + EMPTY_CMD_LINE + EMPTY_LOGIN_NAME + EMPTY_EUID + EMPTY_BINARY_FILE
        return socket_encoding

    @staticmethod
    def create_dataset(name: str,
                       depth: int,
                       history_len: int,
                       degree_dist: np.array,
                       no_of_classes: int,
                       no_of_graphs_per_class: np.array,
                       cmd_line_dist: np.array,
                       login_name_dist: np.array,
                       euid_dist: np.array,
                       binary_file_dist: np.array,
                       node_type_dist: np.array):
        """
        Method that creates a basic 2-Node synthetic dataset

        :param name: name of the dataset
        :param depth: how deep to look in the ancestor graph of a file
        :param history_len: how long to look in the past for interactions with a file
        :param degree_dist: degree distribution of nodes in the graphs
        :param no_of_classes: number of classes of graphs
        :param no_of_graphs_per_class: list containing number of graphs in each class
        :param cmd_line_dist: prob dist of cmd line type (for processes)
        :param login_name_dist: prob dist of login names (for processes)
        :param euid_dist: probability distribution of euids (for processes)
        :param binary_file_dist: probability distribution of provenance binary file (for files)
        :param node_type_dist: [0] - prob of creating a process, [1] - prob of creating a process
        :return: a synthetic dataset that respects the given metrics
        """

        main_dir_path = create_directory(get_directory() + '/Data_Sets/Provenance_Graphs', name)

        for class_number in range(1, no_of_classes + 1):  # for each class
            class_dir_path = create_directory(main_dir_path, 'Class_' + str(class_number))
            clear_directory(class_dir_path)

            property_file = open(main_dir_path + '/property_file', 'a')
            property_file.truncate(0)
            print(no_of_classes, file=property_file)

            for dir_size in no_of_graphs_per_class:
                print(dir_size, file=property_file, end=' ')

            for iterator in range(1, no_of_graphs_per_class[class_number - 1] + 1):  # for each graph in the class
                graph_file = open(class_dir_path + '/provenance_graph_' + str(iterator), 'a')
                graph_file.truncate(0)

                # keep track of how many nodes are generated
                node_counter = 0

                edges = list()
                files = list()

                for file in range(0, history_len):  # file nodes, i.e. different versions of the same file in time
                    node_counter += 1

                    files.append(node_counter)

                    nodes_at_depth = dict()
                    nodes_at_depth['0'] = [node_counter]

                    for level in range(0, depth):  # create ancestors at increasing depth levels
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

                # add edges between file versions
                for index in range(1, len(files)):
                    edges.append((files[index - 1], files[index]))

                for node in range(1, no_of_nodes + 1):
                    if node in files:
                        attribute = SyntheticDataGenerator.__create_file_vector(binary_file_dist[class_number - 1])
                        for value in attribute:
                            print(value, file=graph_file, end=' ')
                        print(file=graph_file)
                    else:
                        # choose to create either socket or file with given probabilities
                        choice = np.random.choice([0, 1], p=node_type_dist)
                        if choice:
                            attribute = SyntheticDataGenerator.__create_socket_vector()
                        else:
                            attribute = SyntheticDataGenerator.__create_process_vector(cmd_line_dist[class_number - 1],
                                                                                       login_name_dist[
                                                                                           class_number - 1],
                                                                                       euid_dist[class_number - 1])
                        for value in attribute:
                            print(value, file=graph_file, end=' ')
                        print(file=graph_file)

                for edge in edges:
                    print(edge[0], edge[1], file=graph_file)
