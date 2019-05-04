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
    import networkx as nx
    graph = nx.DiGraph()

    connection = Neo4JConnection(driver)

    nodes = [
        connection.get_nodes('file'),
        connection.get_nodes('process'),
        connection.get_nodes('socket'),
        connection.get_nodes('pipe')
    ]

    lengths = [
        len(nodes[0]),
        len(nodes[1]),
        len(nodes[2]),
        len(nodes[3])
    ]

    edges = [
        connection.get_edges('INF', 'file', 'process'),
        connection.get_edges('INF', 'process', 'socket'),
        connection.get_edges('INF', 'process', 'file'),
        connection.get_edges('INF', 'socket', 'process'),
        connection.get_edges('INF', 'pipe', 'process'),
        connection.get_edges('INF', 'process', 'pipe')
    ]

    counter = 0
    for node in nodes:
        for vertex in node:
            counter += 1
            graph.add_node(vertex['db_id'])

    print(len(graph.nodes))

    for edge in edges:
        for influence in edge:
            graph.add_edge(influence['parent'], influence['children'])

    graph.remove_nodes_from(list(nx.isolates(graph)))

    for node in graph.nodes:
        print(node, connection.find_name(node))

    return
    summ = 0
    for edge in edges:
        nr = len(edge)
        summ += nr
        print(nr)

    print(summ)

    paths = connection.get_names()
    print(len(paths))
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

    classess = [
        [1417437, 1124392, 254217, 1443042, 26342, 766028, 1387201, 1391198, 127790, 1421946],
        [76294, 1455162, 259003, 921021, 764738, 1384744, 94175, 939128, 686613, 1003905],
        [95854, 1534515, 1583267, 978750, 788682, 967900, 57340, 628462, 8777735, 914895],
        [1124392, 254217, 26342, 1387201, 1391198, 1131273, 1669577, 1421946, 1495211, 842106],
        [1454942, 200197, 937717, 733082, 1421003, 1601232, 1550603, 1115792, 629202, 629666],
        [1045006, 859697, 292949, 635486, 1033777, 144688, 274821, 173179, 638030, 277103]
    ]

    counter = 0
    processes = [[], [], [], [], [], []]

    for nodes in classess:
        for node in nodes:
            bfs = connection.breadth_first_search(node, limit=5, max_depth=5)[1:]
            print(bfs)
            for element in bfs:
                processes[counter].append(element['db_id'])
        counter += 1

    for cls in processes:
        print(390582905832904823905829308590485923856902438690575876089257692385768925367892537698027568932765892475)
        for process in cls:
            props = connection.get_process_attributes(process)
            print(props)


neo4j_driver = Neo4JDriver(
    url='bolt://localhost:7687',
    user='neo4j',
    pswd='opus'
)

get_node_degree_distribution(neo4j_driver)
