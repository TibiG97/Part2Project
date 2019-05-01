from Data_Processing.neo4j_driver import Neo4JDriver, Neo4JConnection
from Data_Processing.graph_structure import Graph
import sys
import networkx as nx
import random


def create_graph(driver: Neo4JDriver):
    local_graph = nx.DiGraph()
    interaction_object = Neo4JConnection(driver)
    files = interaction_object.get_nodes('Store')
    processes = interaction_object.get_nodes('Actor')
    conduits = interaction_object.get_nodes('Conduit')
    edit_sessions = interaction_object.get_nodes('EditSession')

    random_stuff_file = open('results.txt', 'a')
    random_stuff_file.truncate(0)
    commands = list()
    print(len(processes))
    counter = 0

    count_1004 = 0
    count_22 = 0
    count_1002 = 0
    count_1003 = 0
    count_none = 0

    count_steve = 0
    count_nousr = 0
    for process in processes:
        if process['cmd_line'] is not None and process['cmd_line'].find('ssh') != -1:
            commands.append(process['cmd_line'])
            attributes = interaction_object.get_process_attributes(process['db_id'])
            influenced_file = interaction_object.get_influenced_file(process['db_id'])
            print(influenced_file)
            print(influenced_file[0]['n'])
            return
            print(interaction_object.get_context_file(influenced_file[0]['uuid']))
            print(attributes)
            print()
            print()

            if attributes[0]['euid'] == '1004':
                count_1004 += 1
            elif attributes[0]['euid'] == '22':
                count_22 += 1
            elif attributes[0]['euid'] == '1002':
                count_1002 += 1
            elif attributes[0]['euid'] == '1003':
                count_1003 += 1
            else:
                count_none += 1

            if attributes[0]['login_name'] == 'steve':
                count_steve += 1
            else:
                count_nousr += 1

    sum1 = count_nousr + count_steve
    sum2 = count_22 + count_1004 + count_none + count_1002 + count_1003
    print(count_nousr / sum1)
    print(count_steve / sum1)
    print(count_1004 / sum2)
    print(count_1003 / sum2)
    print(count_1002 / sum2)
    print(count_22 / sum2)
    print(count_none / sum2)

    return
    commands.sort()

    for command in commands:
        print(command, file=random_stuff_file)

    sys.exit()

    file_process_edges = interaction_object.get_edges('INF', 'Store', 'Actor')
    process_process_edges = interaction_object.get_edges('INF', 'Actor', 'Actor')
    process_conduit_edges = interaction_object.get_edges('INF', 'Actor', 'Conduit')
    conduit_process_edges = interaction_object.get_edges('INF', 'Conduit', 'Actor')

    for file in files:
        local_graph.add_node(file['db_id'])

    for process in processes:
        local_graph.add_node(process['db_id'])
        print(process['db_id'])
        # print(interaction_object.get_process_attributes(process['db_id']))

    for conduit in conduits:
        local_graph.add_node(conduit['db_id'])

    for edit_session in edit_sessions:
        local_graph.add_node(edit_session['db_id'])

    return local_graph


def build_local_graph_from_neo4j(driver: Neo4JDriver,
                                 partial=False,
                                 node_type1=None,
                                 node_type2=None):
    local_graph = nx.Graph()

    interaction_object = Neo4JConnection(driver)

    files = interaction_object.get_nodes('Store')
    print(len(files))
    processes = interaction_object.get_nodes('Actor')
    print(len(processes))
    file_process_edges = interaction_object.get_edges('INF', 'Store', 'Actor')
    print(len(file_process_edges))
    for file in files:
        local_graph.add_node(file['db_id'])

    print('finished files')
    for process in processes:
        local_graph.add_node(process['db_id'])

    print('finished processes')
    for edge in file_process_edges:
        local_graph.add_edge(edge['parent.db_id'], edge['children.db_id'])

    print('finished edges')
    # print(local_graph.nodes)
    # print(local_graph.edges)

    distribution = list()

    # values = nx.clustering(local_graph)
    # print(values)

    return local_graph

    degree_distribution = open('degree_distribution', 'a')
    css_distribution = open('css_distribution', 'a')
    degree_distribution.truncate(0)
    css_distribution.truncate(0)

    maxdegree = 0
    different_degrees = dict()
    sum = 0

    for degree in local_graph.degree:
        maxdegree = max(maxdegree, degree[1])
        sum += degree[1]
        print(degree[1])
        different_degrees[degree[1]] = 0
        if degree[1] > 0:
            print(degree[1], file=degree_distribution)

    print(sum)

    tanana = list()
    for degree in different_degrees.keys():
        tanana.append(degree)

    tanana.sort()
    for degree in tanana:
        print(str(degree) + ' ' + str(round(random.uniform(0, 0.5), 2)), file=css_distribution)

    print(maxdegree)
    sys.exit()
    for value in values.keys():
        print(values[value], file=css_distribution)

    # print(nx.clustering(local_graph))

    # print(nx.clustering(local_graph))

    sys.exit()

    degrees = interaction_object.get_process_in_degree()
    degrees = sorted(degrees, key=lambda k: k['degree'])
    degrees.reverse()
    print(degrees[0])

    sys.exit()

    processes = interaction_object.get_nodes('Actor')
    files = interaction_object.get_nodes('Store')
    print(1)
    file_process_edges = interaction_object.get_edges('INF', 'Store', 'Actor')
    print(2)
    file_file_edges = interaction_object.get_edges('INF', 'Store', 'Store')
    print(3)
    process_file_edges = interaction_object.get_edges('INF', 'Actor', 'Store')
    print(4)
    process_process_edges = interaction_object.get_edges('INF', 'Actor', 'Actor')
    print(5)

    all_nodes = processes + files
    all_edges = file_process_edges + file_file_edges + process_file_edges + process_process_edges

    print(all_nodes.__len__())
    print(all_edges.__len__())

    for node in all_nodes:
        print(node['db_id'])
        local_graph.add_vertex(node['db_id'])

    print(6)

    for edge in all_edges:
        local_graph.add_edge((edge['parent.db_id'], edge['children.db_id']))

    print(7)

    print(local_graph.nodes())
    print(local_graph.edges())

    return local_graph


def load_data_from_neo4j(driver: Neo4JDriver):
    return




neo4j_driver = Neo4JDriver(
    url='bolt://localhost:7687',
    user='neo4j',
    pswd='opus'
)

# build_local_graph_from_neo4j(neo4j_driver)
# create_graph(neo4j_driver)

create_data()