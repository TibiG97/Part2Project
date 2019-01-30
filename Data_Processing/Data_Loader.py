from Data_Processing.Neo4J_Interaction import Neo4JDriver, Neo4JInteraction
from Data_Processing.Graph_Structure import Graph


def build_local_graph_from_neo4j(driver: Neo4JDriver):
    local_graph = Graph(oriented=True)

    interaction_object = Neo4JInteraction(driver)

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

build_local_graph_from_neo4j(neo4j_driver)
