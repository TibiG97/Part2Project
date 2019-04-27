from utils import get_parent_directory, get_directory
from Data_Processing.Graph_Structure import Graph


def load_local_data_set(name: str,
                        target_model: str):
    """
    Method that loads the local data set stored in the directory called 'name'

    :param name: name of the directory where data set is stored
    :param target_model: decides the format in which to return the data set
    :return: if for patchy_san return all graphs and labels in the data set
            else return all attributes and labels in the data set
    """
    all_graphs = list()
    all_labels = list()

    dataset_directory = get_directory() + '/DataSets/' + name
    number_of_classes, graphs_per_class = load_data_property_file(dataset_directory + '/property_file')

    print(number_of_classes, graphs_per_class)
    for index in range(1, number_of_classes + 1):
        class_directory = dataset_directory + '/Class_' + str(index)
        class_graphs = load_graphs(class_directory, graphs_per_class[index - 1])
        for graph in class_graphs:
            all_graphs.append(graph)
            all_labels.append(index)

    if target_model == 'patchy_san':
        return all_graphs, all_labels
    elif target_model == 'baselines':
        all_values = list()
        for graph in all_graphs:
            all_values.append(graph.values())
        return all_values, all_labels


def load_data_property_file(path: str):
    """
    Method that reads from the file describing the given specific dataset

    :param path: path to the property file
    :return: number of classes and a list of number of graphs per class (in the given dataset)
    """
    graphs_per_class = list()

    property_file = open(path, 'r')
    content = [[int(x) for x in line.split()] for line in property_file]

    number_of_classes = content[0][0]
    for index in range(0, number_of_classes):
        graphs_per_class.append(content[1][index])

    return number_of_classes, graphs_per_class


def load_graphs(directory_path: str,
                number_of_graphs: int):
    """
    Method that loads all provenance graphs from given directory

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