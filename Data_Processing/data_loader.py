from utils import get_directory
from utils import merge_splits
from utils import add_padding
from constants import BIG_NUMBER

from Data_Processing.graph_structure import Graph
from keras_preprocessing.text import hashing_trick

from sklearn.preprocessing import MinMaxScaler

from copy import copy
import numpy as np
import os


class DataLoader:
    """
    Class that implements functionality for loading synthetic data

    """

    @staticmethod
    def load_log_files(dirs):
        """
        :param dirs: list of directories from where to load log files
        :return: log_files dataset and labels for MLP, also the list of all logs' names and paths
        """

        data_set = list()
        labels = list()
        names = list()
        paths = list()

        for iterator in range(0, len(dirs)):
            files = os.listdir(dirs[iterator])
            names.append(files)
            for index in range(len(files)):
                paths.append(dirs[iterator])

            for file in files:
                log_file = open(dirs[iterator] + '/' + file, 'r')

                feature_vector = list()  # feature vector for an entire log file

                lines = log_file.readlines()
                for line in lines:
                    line_vector = hashing_trick(text=line,
                                                n=100000,
                                                hash_function=None,
                                                filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                                lower=True,
                                                split=' ')

                    for element in line_vector:
                        feature_vector.append(float(element))

                data_set.append(feature_vector)
                labels.append(iterator + 1)

        # Make shape of dataset uniform for NN
        ######################################
        min_len = copy(BIG_NUMBER)
        for array in data_set:
            min_len = min(min_len, len(array))

        for iterator in range(0, len(data_set)):
            data_set[iterator] = data_set[iterator][:min_len]
        ######################################

        # use min_max scaling
        scaler = MinMaxScaler()
        data_set = scaler.fit_transform(np.array(data_set))

        names = merge_splits(names)

        return data_set, np.array(labels), names, paths

    @staticmethod
    def load_synthetic_data_set(name, target_model):
        """
        Method that loads the local data set stored in the directory called 'name'

        :param name: name of the directory where data set is stored
        :param target_model: decides the format in which to return the data set
        :return: if for patchy_san return all graphs and labels in the data set
                else return all attributes and labels in the data set
        """

        all_graphs = list()
        all_labels = list()

        dataset_directory = get_directory() + '/Data_Sets/Provenance_Graphs/' + name
        number_of_classes, graphs_per_class = DataLoader.__load_data_property_file(dataset_directory + '/property_file')

        for index in range(1, number_of_classes + 1):
            class_directory = dataset_directory + '/Class_' + str(index)
            class_graphs = DataLoader.__load_graphs(class_directory, graphs_per_class[index - 1])
            for graph in class_graphs:
                all_graphs.append(graph)
                all_labels.append(index)

        all_labels = np.array(all_labels)
        all_graphs = np.array(all_graphs)

        if target_model == 'patchy_san':
            return all_graphs, all_labels, number_of_classes

        elif target_model == 'baselines':

            all_values = list()
            for graph in all_graphs:
                all_values.append(merge_splits(graph.values()))

            all_values = add_padding(all_values, 0)
            all_values = np.array(all_values)

            return all_values, all_labels, number_of_classes

    @staticmethod
    def __load_data_property_file(path: str):
        """
        Private method that reads from the file describing the given specific dataset

        :return: number of classes and a list of number of graphs per class (in the given dataset)
        """

        graphs_per_class = list()

        property_file = open(path, 'r')
        content = [[int(x) for x in line.split()] for line in property_file]

        number_of_classes = content[0][0]
        for index in range(0, number_of_classes):
            graphs_per_class.append(content[1][index])

        return number_of_classes, graphs_per_class

    @staticmethod
    def __load_graphs(directory_path: str,
                      number_of_graphs: int):
        """
        Private method that loads all provenance graphs from given directory

        :param directory_path: path to directory where graphs are stored
        :param number_of_graphs: number of graphs in the directory
        :return: all graphs from the directory in Graph object format
        """

        class_graphs = list()
        for index in range(1, number_of_graphs + 1):
            graph_file = open(directory_path + '/provenance_graph_' + str(index), 'r')
            content = [[int(x) for x in line.split()] for line in graph_file]

            # Computing main properties of interest of each graph
            #####################################################
            no_of_nodes = content[0][0]
            no_of_edges = content[0][1]
            attributes = list()
            edges = list()
            for i in range(1, no_of_nodes + 1):

                # Turn attributes from int to float
                for iterator in range(0, len(content[i])):
                    content[i][iterator] = float(content[i][iterator])

                attributes.append(content[i])

            for i in range(no_of_nodes + 1, no_of_nodes + no_of_edges + 1):
                edges.append((content[i][0], content[i][1]))
            #####################################################

            # Use computer properties to generate the nx.Graph we need
            ##########################################################
            graph = Graph()
            for i in range(1, no_of_nodes + 1):
                graph.add_vertex(i)
                graph.add_one_attribute(i, attributes[i - 1])
            for edge in edges:
                graph.add_edge(edge)
            ##########################################################

            class_graphs.append(graph)

        return class_graphs
