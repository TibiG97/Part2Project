from Data_Processing.neo4j_driver import Neo4JDriver, Neo4JConnection
from Data_Processing.graph_structure import Graph
import sys
import networkx as nx
import random


def test(driver: Neo4JDriver):
    connection = Neo4JConnection(driver)

    names = connection.get_names()

    for iterator in range(0, len(names)):
        names[iterator]['path'] = names[iterator]['path'].split('/')[:3]
        file = connection.get_file_named(names[iterator]['dbid'])
        print(names[iterator])
        print(file)
        print(connection.get_node_outdegree(file[0]['dbid']))

    classes = [[], [], [], [], [], []]
    L = 10
    for name in names:
        if len(name['path']) > 2:
            if name['path'][2] == 'ports' and len(classes[0]) < L:
                classes[0].append(name['dbid'])
            if name['path'][2] == 'freebsd' and len(classes[1]) < L:
                classes[1].append(name['dbid'])
            if name['path'][2] == 'obj' and len(classes[2]) < L:
                classes[2].append(name['dbid'])
            if name['path'][2] == 'src' and len(classes[3]) < L:
                classes[3].append(name['dbid'])
            if name['path'][2] == 'starc' and len(classes[4]) < L:
                classes[4].append(name['dbid'])
            if name['path'][2] == 'local' and len(classes[5]) < L:
                classes[5].append(name['dbid'])

    counter = 0
    for cls in classes:
        for node in cls:
            file = connection.get_file_named(node)
            bfs = connection.breadth_first_search(file[0]['dbid'], limit=3, max_depth=3)
            print(bfs)
            counter += 1
            print(counter)


def load_graph(driver: Neo4JDriver):
    graph = Graph()

    connection = Neo4JConnection(driver)
    print(len(connection.get_all_edges()))
    return
    edges = [
        connection.get_edges('INF', 'file', 'process'),
        connection.get_edges('INF', 'file', 'socket'),
        connection.get_edges('INF', 'file', 'pipe'),
        connection.get_edges('INF', 'file', 'file'),
    ]
    paths = connection.get_names()
    files = connection.get_nodes(node_type='file')
    processes = connection.get_nodes(node_type='process')
    return graph


def get_node_type_distribution(driver: Neo4JDriver):
    connection = Neo4JConnection(driver)

    files = connection.get_nodes(node_type='file')
    processes = connection.get_nodes(node_type='process')
    sockets = connection.get_nodes(node_type='socket')
    pipes = connection.get_nodes(node_type='pipe')

    return {
        'files': len(files),
        'processes': len(processes),
        'sockets': len(sockets),
        'pipes': len(pipes),
        'nodes': len(files) + len(processes) + len(sockets) + len(pipes)
    }


def get_node_degree_distribution(driver: Neo4JDriver):
    connection = Neo4JConnection(driver)

    nodes = connection.get_nodes()


neo4j_driver = Neo4JDriver(
    url='bolt://localhost:7687',
    user='neo4j',
    pswd='opus'
)

load_graph(neo4j_driver)
