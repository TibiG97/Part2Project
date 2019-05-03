from Data_Processing.neo4j_driver import Neo4JDriver, Neo4JConnection
from Data_Processing.graph_structure import Graph
import sys
import networkx as nx
import random


def test(driver: Neo4JDriver):
    connection = Neo4JConnection(driver)

    print(connection.breadth_first_search(135, 5, 5))


neo4j_driver = Neo4JDriver(
    url='bolt://localhost:7687',
    user='neo4j',
    pswd='opus'
)

test(neo4j_driver)
