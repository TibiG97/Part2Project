import networkx as nx


class Graph(object):

    def __init__(self, oriented=False, label=None):
        if oriented:
            self.nx_graph = nx.DiGraph()
        else:
            self.nx_graph = nx.Graph()
        self.label = label

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
