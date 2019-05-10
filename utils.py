from random import shuffle

from os import pardir
from os import getcwd
from os import makedirs
from os import listdir
from os import unlink

from os.path import abspath
from os.path import join
from os.path import isdir
from os.path import isfile

from shutil import rmtree

import numpy as np

import networkx as nx


def get_directory():
    """
    :return: Path to directory from which the function is executed
    """
    return getcwd()


def get_parent_directory():
    """
    :return: Path to the parent of the directory from which the function is executed
    """
    return abspath(join(getcwd(), pardir))


def create_directory(father_dir_path, name):
    """
    Functions that creates a new directory at a specified path

    :param father_dir_path: path where we want to create a new directory
    :param name: name of the directory we want to create
    :return: path to the created directory
    """
    dir_path = father_dir_path + '/' + name
    if not isdir(dir_path):
        makedirs(dir_path)
    else:
        rmtree(dir_path)

    return dir_path


def clear_directory(dir_path):
    """
    Function that delets the files of a given directory

    :param dir_path: path to the directory we are interested in
    """
    for file in listdir(dir_path):
        file_path = join(dir_path, file)
        if isfile(file_path):
            unlink(file_path)

    return


def randomise_order(list_x: np.array,
                    list_y: np.array):
    """
    Function that randomises X,Y lists according with the same permutation

    :param list_x: first list
    :param list_y: second list
    :return: simmetrically randomised X,Y lists and the permutation of the randomisation
    """

    permutation = range(0, len(list_x))
    merged_data_set = list(zip(list_x, list_y, permutation))
    shuffle(merged_data_set)
    list_x, list_y, permutation = zip(*merged_data_set)

    return np.array(list_x), np.array(list_y), list(permutation)


def randomise_order4(list1: np.array,
                     list2: np.array,
                     list3: np.array,
                     list4: np.array):
    """
    Function that randomises order of 4 lists in sync

    :return: the 4 shuffled lists
    """

    merged_data_set = list(zip(list1, list2, list3, list4))
    shuffle(merged_data_set)
    list1, list2, list3, list4 = zip(*merged_data_set)

    return np.array(list1), np.array(list2), np.array(list3), np.array(list4)


def split_in_folds(data: np.array,
                   no_of_splits: int):
    """
    Function that splits a data list into a specified number of folds

    :param data: dataset to be splitted
    :param no_of_splits: specified number of splits
    :return: A list of required number of sublists representing the mentioned splits
    """

    splits = list()
    size = len(data)
    for iterator in range(0, no_of_splits):
        current_split = list()
        left = int(size / no_of_splits) * iterator
        right = int(size / no_of_splits) * (iterator + 1)
        for indelist_x in range(left, right):
            current_split.append(data[indelist_x])
        splits.append(current_split)
    return np.array(splits)


def merge_splits(data: np.array):
    """
    Function that merges a list of lists representing the folds

    :param data: list of lists to be merged
    :return: a single list containing all elements in all folds
    """

    merged_list = list()
    for split in data:
        for element in split:
            merged_list.append(element)
    return np.array(merged_list)


def convert_labels_to_pos_neg(labels: np.array):
    """
    Function that makes input suitable for ROC_CURVE and PRECISION_RECALL_CURVE functions

    :param labels: A list of labels for 2 classes (1s and 2s)
    :return: A where 2s are replaced by 0s
    """

    new_labels = list()
    for element in labels:
        if element == 1:
            new_labels.append(1)
        else:
            new_labels.append(-1)

    return np.array(new_labels)


def add_padding(matrix: list,
                value: int):
    """
    :param matrix: matrix to be padded
    :param value: value to pad with
    :return: padded matrix
    """

    len_max = 0
    for line in matrix:
        len_max = max(len_max, len(line))

    for iterator in range(0, len(matrix)):
        if len(matrix[iterator]) < len_max:
            matrix[iterator] = np.concatenate((matrix[iterator],
                                               [value] * (len_max - len(matrix[iterator]))), axis=None)

    return matrix


def create_ensamble(predictions1: np.array,
                    predictions2: np.array,
                    labels: np.array,
                    meta_classifier,
                    no_of_folds: int):
    """
    Function that cross validates an ensamble of two classifiers using stacking method

    :param predictions1: classifier #1 predictions
    :param predictions2: classifier #2 predictions
    :param labels: true labels corresponding to predictions
    :param meta_classifier: metaclassifier of stacking method
    :param no_of_folds: folds for outer CV of stacker
    :return: resulyts of the stacked classifier
    """

    dataset = np.array([np.array([pred1, pred2]) for pred1, pred2 in zip(predictions1, predictions2)])
    lrg_acc, lrg_all_acc, lrg_pred, h = meta_classifier.cross_validate(dataset, labels, no_of_folds=no_of_folds,
                                                                       clear_file=True)

    return lrg_acc, lrg_all_acc, lrg_pred, h


def compute_ranking_distance(graph, vertex):
    """
    Function that computes all distances of nodes in a graph
    from a given node using dijkstra's algorithm

    :param graph: graph in which to compute distances
    :param vertex: landmark node to which we compute distances
    :return: graph labeled w.r.t. the computed distances
    """

    labeled_graph = nx.Graph(graph)
    source_path_lengths = nx.single_source_dijkstra_path_length(graph, vertex)
    nx.set_node_attributes(labeled_graph, source_path_lengths, 'labeling')

    return labeled_graph


def betweenness_centrality(graph: nx.Graph):
    """
    Function that labels a graph w.r.t. the values of betweenness centrality on each node

    :param graph: graph to be labeled
    :return: ordering dictionaries
    """

    labeling_procedure = {}
    labeled_graph = nx.Graph(graph)

    centrality = list(nx.betweenness_centrality(graph).items())
    centrality = sorted(centrality, key=lambda n: n[1], reverse=True)

    dictionary = {}
    label = 0

    for t in centrality:
        dictionary[t[0]] = label
        label += 1

    nx.set_node_attributes(labeled_graph, dictionary, 'labeling')
    ordered_nodes = list(zip(*centrality))[0]

    labeling_procedure['labeled_graph'] = labeled_graph
    labeling_procedure['sorted_centrality'] = centrality
    labeling_procedure['ordered_nodes'] = ordered_nodes

    return labeling_procedure


def node_timestamp(graph: nx.Graph):
    """
    Function that labels a graph w.r.t. the values of nodes' timestamps
    
    :param graph: graph to be ordered 
    :return: ordering dictionaries
    """

    labeling_procedure = {}
    labeled_graph = nx.Graph(graph)
    timestamps = list(nx.betweenness_centrality(graph).items())

    counter = 0
    for iterator in range(len(labeled_graph.nodes)):
        timestamps[iterator] = (timestamps[iterator][0], counter)
        counter += 1

    timestamps = sorted(timestamps, key=lambda n: n[1], reverse=False)

    dictionary = {}
    label = 0

    for t in timestamps:
        dictionary[t[0]] = label
        label += 1

    nx.set_node_attributes(labeled_graph, dictionary, 'labeling')
    ordered_nodes = list(zip(*timestamps))[0]

    labeling_procedure['labeled_graph'] = labeled_graph
    labeling_procedure['sorted_centrality'] = timestamps
    labeling_procedure['ordered_nodes'] = ordered_nodes

    return labeling_procedure


def relabel_graph(graph: nx.Graph,
                  original_labeling: dict,
                  new_labeling: dict):
    """
    Method that relabels a graph given a new dictionary of labels

    :param graph: graph to be relabeled
    :param original_labeling: initial label dictionary
    :param new_labeling: new label dictionary
    :return: new order of the nodes in the relabeled graph
    """

    labels = list(set(original_labeling.values()))
    new_order = original_labeling
    max_label = 0

    for label in labels:

        nodes_of_label_label = [x for x, y in graph.nodes(data=True) if y['labeling'] == label]

        if len(nodes_of_label_label) >= 2:
            inside_ordering = sorted(nodes_of_label_label, key=new_labeling.get)
            inside_order = dict(zip(inside_ordering, range(len(inside_ordering))))

            for k, v in inside_order.items():
                new_order[k] = max_label + 1 + inside_order[k]

            max_label = max_label + len(nodes_of_label_label)

        else:
            new_order[nodes_of_label_label[0]] = max_label + 1
            max_label = max_label + 1

    return new_order
