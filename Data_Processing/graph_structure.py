import networkx as nx


class Graph(object):

    def __init__(self, oriented=False):
        if oriented:
            self.nx_graph = nx.DiGraph()
        else:
            self.nx_graph = nx.Graph()

    def nodes(self):
        """
        :return: Dictionary of nodes in the graph
        """

        return dict(self.nx_graph.nodes())

    def edges(self):
        """
        :return: List of edges in the graph
        """

        return self.nx_graph.edges()

    def add_vertex(self,
                   vertex: int):
        """
        Function that adds one verted to the graph

        :param vertex: id of vertex to be added
        """

        if vertex not in self.nodes():
            self.nx_graph.add_node(vertex)

    def values(self):
        """
        :return: list of all attributes of all nodes
        """

        return [v for (k, v) in nx.get_node_attributes(self.nx_graph, 'attr_name').items()]

    def add_nodes(self,
                  nodes):
        """
        Function that adds one or more vertices to the graph

        :param nodes: list of nodes to be added
        """

        self.nx_graph.add_nodes_from(nodes)

    def add_edge(self, edge):
        """
        Function that adds an edge in the graph

        :param edge: (edge1, edge2)
        """

        (vertex1, vertex2) = tuple(edge)
        self.nx_graph.add_edge(vertex1, vertex2)

    def add_one_attribute(self,
                          node: int,
                          attr):
        """
        :param node: node to which to add the atrribute
        :param attr: attribute to be added
        """

        self.nx_graph.add_node(node, attr_name=attr)

    def get_attributes(self, vertex):
        """
        :param vertex: id of the vertex for which to recover attributes
        """

        return self.nx_graph.node[vertex]
