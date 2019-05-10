import os
import numpy as np

from constants import CMD_LINE_DICT
from constants import LOGIN_NAME_DICT
from constants import LOG_DIRS
from constants import CLASS_DICT

from pickle import load

from Data_Processing.data_loader import DataLoader


def pos_one_hot(vector: np.array):
    """
    :param vector: one hot encoded feature vector
    :return: position where 1 is in the vector
    """

    for iterator in range(len(vector)):
        if vector[iterator] == 1.0:
            return iterator


def get_properties(feature_vector: np.array):
    """
    :param feature_vector: feature vector of a process
    :return: returns cmd_line, login_name of a process as a dictionary
    """

    cmd_line = CMD_LINE_DICT[pos_one_hot(feature_vector[3:15])]
    login_name = LOGIN_NAME_DICT[pos_one_hot(feature_vector[15:21])]

    return {
        'cmd_line': cmd_line,
        'login_name': login_name
    }


def rename():
    """
    Function that renames the files using classification results and user's inputs

    """

    graph_dataset, graph_labels, no_of_classes = DataLoader.load_synthetic_data_set(name='D3_H4_L02',
                                                                                    target_model='patchy_san')
    X, y, names, paths = DataLoader.load_log_files(LOG_DIRS)

    predictions = load(open('Dumped/lrg_pred', 'rb'))
    params = ['cmd_line', 'login_name', 'class', 'timestamp']

    # get the parameters that user is interested in
    ###############################################
    ###############################################

    correct_input = False
    user_params = None
    no_of_processes = None

    while not correct_input:
        user_params = input('Please enter your renaming strategy, '
                            'you can choose one or more of the '
                            'following parameters to use for the new names:'
                            + '\n' + str(params) + '\n' +
                            'Please add a comma after each chosen parameter' + '\n')

        user_params = user_params.split(',')
        for iterator in range(len(user_params)):
            user_params[iterator] = user_params[iterator].strip()

        print(user_params)
        no_of_processes = input('Now choose how many processes you want to take into account (max 5):' + '\n')
        no_of_processes = int(no_of_processes)

        correct_input = True
        for prm in user_params:
            if prm not in params:
                correct_input = False
            if not 1 <= no_of_processes <= 5:
                correct_input = False

        if not correct_input:
            print('You have introduces the parameters wrongly, please try again' + '\n')
    ###############################################
    ###############################################

    # extract properties required by the user from the graphs
    # and create new names using the selection of properties
    #########################################################
    #########################################################

    proposals_file = open('new_names', 'a')
    proposals_file.truncate(0)
    new_names = list()

    for it in range(len(predictions)):

        nodes = graph_dataset[it].nodes()
        all_properties = list()

        selected = 0
        counter = 2

        while selected < no_of_processes:
            feature_vector = nodes[counter]['attr_name']

            if feature_vector[1] == 1.0:
                selected += 1
                properties = get_properties(feature_vector)
                properties['timestamp'] = counter
                all_properties.append(properties)

            counter += 1

        new_name = str()
        if 'class' in user_params:
            new_name += CLASS_DICT[graph_labels[it]] + '_'
        for iterator in range(no_of_processes):
            if 'cmd_line' in user_params:
                new_name += all_properties[iterator]['cmd_line'] + '_'
            if 'login_name' in user_params:
                new_name += all_properties[iterator]['login_name'] + '_'
        new_name += names[it].split('_')[-1]

        new_names.append(new_name)

        print(names[it], ' -----> ', new_name, file=proposals_file)

    #########################################################
    #########################################################

    # ask user to confirm the renaming strategy
    ###########################################
    ###########################################

    correct_input = False
    user_response = None
    while not correct_input:

        user_response = input('The renaming proposals can be found in file: new_names' + '\n' +
                              'write OK if you agree with them or CANCEL if you want to discard them' + '\n')

        correct_input = True
        if user_response != 'OK' and user_response != 'CANCEL':
            correct_input = False
            print('You have introduces the command wrongly, please try again' + '\n')

    ###########################################
    ###########################################

    if user_response == 'CANCEL':
        print('The files names remain unchanged')
        return

    for iterator in range(len(new_names)):
        os.rename(paths[iterator] + '/' + names[iterator], paths[iterator] + '/' + new_names[iterator])
