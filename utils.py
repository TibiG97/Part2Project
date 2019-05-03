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
    :return: simmetrically randomised X,Y lists
    """
    merged_data_set = list(zip(list_x, list_y))
    shuffle(merged_data_set)
    list_x, list_y = zip(*merged_data_set)
    return np.array(list_x), np.array(list_y)


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
