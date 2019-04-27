from random import shuffle


def randomise_order(list_x, list_y):
    """
    Method that randomises X,Y lists according with the same permutation
    :param list_x: first list
    :param list_y: second list
    :return: simmetrically randomised X,Y lists
    """
    merged_data_set = list(zip(list_x, list_y))
    shuffle(merged_data_set)
    list_x, list_y = zip(*merged_data_set)
    return list_x, list_y


def split_in_folds(data: list,
                   no_of_splits: int):
    """
    Method that splits a data list into a specified number of folds

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
    return splits


def merge_splits(data: list):
    """
    Method that merges a list of lists representing the folds

    :param data: list of lists to be merged
    :return: a single list containing all elements in all folds
    """
    merged_list = list()
    for split in data:
        for element in split:
            merged_list.append(element)
    return merged_list
