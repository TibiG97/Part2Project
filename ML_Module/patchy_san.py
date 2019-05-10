from networkx import nx
from networkx import convert_node_labels_to_integers
from pynauty.graph import canonical_labeling, Graph

from utils import compute_ranking_distance
from utils import betweenness_centrality
from utils import node_timestamp
from utils import relabel_graph

import numpy as np


class PatchySAN(object):
    def __init__(self,
                 graph: nx.Graph,
                 width: int,
                 stride: int,
                 rf_size: int,
                 dummy_value: np.array,
                 labeling_procedure_name='betweenness'):
        """
        Constructor of PatchySAN class

        :param graph: input graph
        :param width: number of receptive fields to create
        :param stride: distance between BFS starting nodes w.r.t labeling
        :param rf_size: receptive field size
        :param dummy_value: value to add as dummy node
        :param labeling_procedure_name: betweenness centrality or node timestamps
        """

        self.graph = graph
        self.width = width
        self.stride = stride
        self.rf_size = rf_size
        self.dummy_value = dummy_value
        self.labeling_procedure_name = labeling_procedure_name
        self.initial_labeling = self.labeling_function(self.graph)

    def create_all_rfs(self):
        """
        Method that transforms the graph attribute of the PCSN object into suitable input for CNN

        :return: (width, rf_size, attr_dim) input for CNN
        """

        input_to_cnn = list()

        # select node sequence returns full list of receptive fields created
        receptive_fields = self.node_sequence_selection()

        for field in receptive_fields:
            relabeled_nodes = nx.relabel_nodes(field, nx.get_node_attributes(field, 'labeling'))

            input_to_cnn.append(
                [x[1] for x in sorted(nx.get_node_attributes(relabeled_nodes, 'attr_name').items(),
                                      key=lambda x: x[0])])

        return input_to_cnn

    def node_sequence_selection(self):
        """
        Method that uses the initial ordering of nodes and calles receptive_field_maker within given strides
        
        :return: a 'width' number of receptive fields, some of them perhaps with dummy values 
        """

        sorted_vertices = self.initial_labeling['ordered_nodes']
        receptive_fields = []
        iterator_1 = 0
        iterator_2 = 1

        while iterator_2 <= self.width:

            if iterator_1 < len(sorted_vertices):
                receptive_fields.append(
                    self.create_receptive_field(sorted_vertices[iterator_1])
                )

            else:
                receptive_fields.append(
                    self.create_empty_receptive_field()
                )

            iterator_1 += self.stride
            iterator_2 += 1

        return receptive_fields

    def neighborhood_assembly(self, vertex):
        """
        Method that conducts a BFS, establishing a neighbourhood of nodes of given size

        :param vertex: starting node in the BFS
        :return: subgraph obtained by the BFS
        """

        all_nodes_in_bfs = {vertex}
        direct_neighbours = {vertex}
        while len(all_nodes_in_bfs) < self.rf_size \
                and len(direct_neighbours) > 0:

            neighbours = set()
            for neighbour in direct_neighbours:
                neighbours = neighbours.union(set(self.graph.neighbors(neighbour)))

            direct_neighbours = neighbours - all_nodes_in_bfs

            all_nodes_in_bfs = all_nodes_in_bfs.union(direct_neighbours)

        return self.graph.subgraph(list(all_nodes_in_bfs))

    def receptive_field_normalization(self,
                                      graph: nx.graph,
                                      vertex: int):
        """
        Method that normalizes a provenance graph

        :param graph: provenance graph to be normalized
        :param vertex: starting vertex in the neighbourhood assembly, from which to compute the ranking function
        :return: normalized (and canonicalized) provenance graph
        """

        ranked_graph = self.labeling_function(graph)['labeled_graph']
        original_order = nx.get_node_attributes(ranked_graph, 'labeling')

        graph_subset = self.compute_graph_ranking(graph,
                                                  vertex,
                                                  original_order)

        if len(graph_subset.nodes()) > self.rf_size:

            d = dict(nx.get_node_attributes(graph_subset, 'labeling'))
            k_first_nodes = sorted(d, key=d.get)[0:self.rf_size]
            full_graph = graph_subset.subgraph(k_first_nodes)

            ranked_graph_by_labeling_procedure = self.labeling_function(graph)['labeled_graph']
            original_order = nx.get_node_attributes(ranked_graph_by_labeling_procedure, 'labeling')
            full_ranked_graph = self.compute_graph_ranking(full_graph, vertex, original_order)

        elif len(graph_subset.nodes()) < self.rf_size:

            full_ranked_graph = self.receptive_field_padding(graph_subset)
        else:

            full_ranked_graph = graph_subset

        return self.nauty_graph_automorphism(full_ranked_graph)

    def labeling_function(self, graph):
        """
        Function that labels a graph w.r.t. labeling procedure input

        :param graph: graph to be labeled
        :return: labeled graph
        """

        if self.labeling_procedure_name == 'betweenness':
            return betweenness_centrality(graph)
        elif self.labeling_procedure_name == 'timestamps':
            return node_timestamp(graph)

    def create_receptive_field(self, vertex):
        """
        Method that creates a receptive field from a given vertex

        :param vertex: start vertex
        :return: receptive field of vertex
        """

        graph = self.neighborhood_assembly(vertex)

        normalized_graph = self.receptive_field_normalization(graph, vertex)

        return normalized_graph

    def create_empty_receptive_field(self):
        """
        Method that creates a dummy receptive field for padding purposes

        :return: a receptive field of dummy nodes
        """

        graph = nx.star_graph(self.rf_size - 1)
        nx.set_node_attributes(graph, self.dummy_value, 'attr_name')
        nx.set_node_attributes(graph, {element: element for element, v in dict(graph.nodes()).items()}, 'labeling')

        return graph

    def receptive_field_padding(self, normalized_graph):
        """
        Method that ensures uniformity across receptive fields when width or rf_size are too big

        :param normalized_graph: rf_transformed graph to which we add dummy nodes
        :return: uniformized graph
        """

        graph = nx.Graph(normalized_graph)
        keys = [key for key, v in dict(normalized_graph.nodes()).items()]
        labels = [value for key, value in dict(nx.get_node_attributes(normalized_graph, 'labeling')).items()]

        # add extra dummy nodes as long as rf_size is not reached
        #########################################################
        counter = 1
        while len(graph.nodes()) < self.rf_size:
            graph.add_node(max(keys) + counter,
                           attr_name=self.dummy_value,
                           labeling=max(labels) + counter)
            counter += 1
        #########################################################

        return graph

    @staticmethod
    def compute_graph_ranking(graph: nx.Graph,
                              vertex: int,
                              original_node_order: dict):
        """
        Method that relabels a graph w.r.t. nodes distances to given root

        :param graph: subgraph to rank
        :param vertex: landmark vertex for the ranking
        :param original_node_order: original ranking
        :return: graph labeled by the new ranking
        """

        labeled_graph = nx.Graph(graph)
        ordered_graph = compute_ranking_distance(graph, vertex)
        labels = nx.get_node_attributes(ordered_graph, 'labeling')

        new_order = relabel_graph(graph=ordered_graph,
                                  original_labeling=labels,
                                  new_labeling=original_node_order)

        nx.set_node_attributes(labeled_graph, new_order, 'labeling')

        return labeled_graph

    @staticmethod
    def nauty_graph_automorphism(graph: nx.Graph):
        """
        Graph canonicalization funtion, meant to break timebreakers of the non-injective ranking function

        :param graph: subgraph to be canonicalized
        :return: canonicalized subgraph
        """

        # convert labels to integers to give nauty the node partitions required
        graph_int_labeled = convert_node_labels_to_integers(graph)
        canonicalized_graph = nx.Graph(graph_int_labeled)

        # get canonicalized graph using nauty
        nauty = Graph(len(graph_int_labeled.nodes()), directed=False)
        nauty.set_adjacency_dict({node: list(nbr) for node, nbr in graph_int_labeled.adjacency()})

        labels_dict = nx.get_node_attributes(graph_int_labeled, 'labeling')
        canonical_labeling_order = {k: canonical_labeling(nauty)[k] for k in
                                    range(len(graph_int_labeled.nodes()))}

        canonical_order = relabel_graph(graph_int_labeled, labels_dict, canonical_labeling_order)
        nx.set_node_attributes(canonicalized_graph, canonical_order, 'labeling')

        return canonicalized_graph
