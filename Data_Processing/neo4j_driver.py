from neo4j.v1 import GraphDatabase


class Neo4JDriver(object):
    """
    Class representing the connection tool with the Neo4J database

    """

    def __init__(self,
                 url: str,
                 user: str,
                 pswd: str):
        """
        Object Constructor

        :param url: address where Neo4J server runs
        :param user: username used for authentication
        :param pswd: password used for authentication
        """

        self._driver = GraphDatabase.driver(url, auth=(user, pswd), encrypted=False)

    def execute_query(self,
                      query: str):
        """
        :param query: query for the database in Cypher format
        :return: answer to query as a list of dictionaries
        """

        session = self._driver.session()

        query_answer = session.run(query)
        records_answer = query_answer.records()

        query_answer = list()
        for record in records_answer:
            query_answer.append(dict(record.items()))

        session.close()

        return query_answer

    def close_connection(self):
        """
        Method that ends connection to the database

        """

        self._driver.close()


class Neo4JConnection(object):
    """
    Class that implements feature extraction functionality

    """

    def __init__(self,
                 driver: Neo4JDriver):
        self._driver = driver

    def get_nodes(self,
                  node_type=None):
        """
        :param node_type: type of nodes to be returned
        :return: all nodes of node_type if not None
        all nodes in the database otherwise
        """

        if node_type is None:
            q = "match (n) return n.db_id as db_id"
        else:
            q = "match (n: %s) return n.db_id as db_id, n.cmdline as cmd_line" % node_type
        answer = self._driver.execute_query(q)

        return answer

    def get_number_of_nodes(self,
                            node_type=None):
        """
        :param node_type: type of nodes to be counted
        :return: number of nodes of node_type if not None
        number of nodes in all database otherwise
        """

        if node_type is None:
            q = "match (n) return count(*)"
        else:
            q = "match (n: %s) return count(*)" % node_type
        answer = self._driver.execute_query(q)

        return answer

    def get_node_indegree(self,
                          uuid: str):
        """
        :param uuid: unique identifier of the node
        :return: in_degree of the node
        """

        q = "match p = ()-[r:INF]->(n {uuid: '%s'}) return count(*) as files_in_touch" % uuid
        answer = self._driver.execute_query(q)
        return answer

    def get_node_outdegree(self,
                           uuid: str):
        """
        :param uuid: unique identifier of the node
        :return: out_degree of the node
        """

        q = "match p = (n {uuid: '%s'})-[r:INF]->() return count(*) as files_in_touch" % uuid
        answer = self._driver.execute_query(q)
        return answer

    def get_edges(self,
                  rel_type: str,
                  node_type1: str,
                  node_type2: str):
        """
        :param rel_type: type of relation we are interested in (INF or NAMED)
        :param node_type1: type of first node
        :param node_type2: type of second node
        :return: all specified relation edges between of any two nodes of given types
        """

        q = "match p = (parent: %s)-[r: %s]->(children: %s) " \
            "return parent.db_id, children.db_id" % (
                node_type1, rel_type, node_type2)

        answer = self._driver.execute_query(q)
        return answer

    def get_number_of_edges(self,
                            rel_type: str,
                            node_type1: str,
                            node_type2: str):
        """
        :param rel_type: type of relation we are interested in (INF or NAMED)
        :param node_type1: type of first node
        :param node_type2: type of second node
        :return: number of specified relation edges between of any two nodes of given types
        """

        q = "match p = (parent: %s)-[r: %s]->(children: %s) " \
            "return count(*)" % (
                node_type1, rel_type, node_type2)

        answer = self._driver.execute_query(q)
        return answer

    def get_process_attributes(self,
                               uuid: str):
        """
        :param uuid: unique identifier of the process node
        :return: all atributes of the process
        """

        q = "match (n: Actor) where n.db_id = %s return " \
            "n.euid as euid, " \
            "n.rgid as rgid, " \
            "n.pid as pid, " \
            "n.suid as suid, " \
            "n.egid as egid, " \
            "n.sgid as sgid, " \
            "n.db_id as db_id, " \
            "n.cmdline as cmd_line, " \
            "n.login_name as login_name" % uuid

        answer = self._driver.execute_query(q)
        return answer

    def get_influenced_file(self,
                            uuid: str):
        q = "match (n:Store)-[r:INF]->(p:Actor {db_id: %d}) return n limit 1" % uuid

        answer = self._driver.execute_query(q)
        return answer

    def get_context_file(self,
                         uuid: str):

        q = "match (n:StoreCont {uuid: %s}) return n" % uuid
        answer = self._driver.execute_query(q)
        return answer
