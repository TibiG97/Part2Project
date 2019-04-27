import networkx as nx

import time


class NoAttrMatrix(Exception):
    pass


class NoPathException(Exception):
    pass


class Graph(object):

    def __init__(self, oriented=False, label=None):
        if oriented:
            self.nx_graph = nx.DiGraph()
        else:
            self.nx_graph = nx.Graph()
        self.label = label

    def __eq__(self, other):
        # print('yo method')
        return self.nx_graph == other.nx_graph

    def __hash__(self):
        return hash(str(self))

    def nodes(self):
        return dict(self.nx_graph.nodes())

    def edges(self):
        return self.nx_graph.edges()

    def add_vertex(self, vertex):
        if vertex not in self.nodes():
            self.nx_graph.add_node(vertex)

    def values(self):
        return [v for (k, v) in nx.get_node_attributes(self.nx_graph, 'attr_name').items()]

    def add_nodes(self, nodes):
        self.nx_graph.add_nodes_from(nodes)

    def add_edge(self, edge):
        (vertex1, vertex2) = tuple(edge)
        self.nx_graph.add_edge(vertex1, vertex2)

    def add_one_attribute(self, node, attr, attr_name='attr_name'):
        self.nx_graph.add_node(node, attr_name=attr)

    def add_attibutes(self, attributes):
        attributes = dict(attributes)
        for node, attr in attributes.items():
            self.add_one_attribute(node, attr)

    def get_attr(self, vertex):
        return self.nx_graph.node[vertex]

    def find_leaf(self, beginwith):  # assez nulle comme recherche
        nodes = self.nodes()
        returnlist = list()
        for nodename in nodes:
            if str(nodename).startswith(beginwith):
                returnlist.append(nodename)
        return returnlist

    def smallest_path(self, start_vertex, end_vertex):
        try:
            pathtime = time.time()
            shtpath = nx.shortest_path(self.nx_graph, start_vertex, end_vertex)
            endpathtime = time.time()
            self.log['pathtime'].append(endpathtime - pathtime)
            return shtpath
        except nx.exception.NetworkXNoPath:
            raise NoPathException('No path between two nodes, graph name : ', self.name)
