from neo4j.v1 import GraphDatabase
from queue import LifoQueue


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
            q = "match (n) where n.ty = '%s' return n.db_id as db_id" % node_type
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
            q = "match (n) where n.ty = '%s' return count(*) as count" % node_type
        answer = self._driver.execute_query(q)

        return answer

    def get_node_indegree(self,
                          dbid: int):
        """
        :param dbid: unique identifier of the node
        :return: in_degree of the node
        """

        q = "match p = ()-[r:INF]->(n {dbid: %d}) return count(*) as files_in_touch" % dbid
        answer = self._driver.execute_query(q)
        return answer

    def get_node_outdegree(self,
                           dbid: int):
        """
        :param dbid: unique identifier of the node
        :return: out_degree of the node
        """

        q = "match p = (n {dbid: %d})-[r:INF]->() return count(*) as files_in_touch" % dbid
        answer = self._driver.execute_query(q)
        return answer

    def get_node_type(self,
                      dbid: int):
        """
        :param dbid: unique identifier of the node
        :return: type of the node with uuid
        """

        q = "match (n {db_id: %d}) return n.ty as type" % dbid
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

        q = "match p = (parent)-[r: %s]->(children) " \
            "where parent.ty = '%s' and children.ty = '%s' " \
            "return parent.db_id as parent, children.db_id as children" % (
                rel_type, node_type1, node_type2)

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

        q = "match p = (parent)-[r: %s]->(children) " \
            "where parent.ty = '%s' and children.ty = '%s' " \
            "return count(*) as count" % (
                rel_type, node_type1, node_type2)

        answer = self._driver.execute_query(q)
        return answer

    def find_name(self,
                  dbid: int):
        """
        :param dbid: unique identifier of the node
        :return: db_id of a node containing the path and name of the file
        """
        q = "match (n) where n.db_id = %d return n.uuid as uuid" % dbid
        answer = self._driver.execute_query(q)

        if len(answer) == 0:
            return None

        q = "match (n:StoreCont) where n.uuid = '%s' return n.db_id as dbid" % answer[0]['uuid']
        answer = self._driver.execute_query(q)

        if len(answer) == 0:
            return None

        q = "match p = (n:StoreCont)-[r:NAMED]->(m:Path) where n.db_id = %d return m.path as path limit 1" % answer[0][
            'dbid']
        answer = self._driver.execute_query(q)

        return answer

    def get_process_attributes(self,
                               dbid: int):
        """
        :param dbid: unique identifier of the process node
        :return: all atributes of the process
        """

        q = "match (n: Actor) where n.db_id = %d return " \
            "n.euid as euid, " \
            "n.rgid as rgid, " \
            "n.pid as pid, " \
            "n.suid as suid, " \
            "n.egid as egid, " \
            "n.sgid as sgid, " \
            "n.db_id as db_id, " \
            "n.cmdline as cmd_line, " \
            "n.login_name as login_name" % dbid

        answer = self._driver.execute_query(q)
        return answer

    def get_context_file(self,
                         dbid: int):
        """
        :param dbid: unique identifier of the node of interest
        :return: file where timestamp is stored
        """

        q = "match (n:StoreCont {dbid: %d}) return n" % dbid
        answer = self._driver.execute_query(q)
        return answer

    def get_neighbours(self,
                       dbid: int,
                       limit: int):
        """
        :param dbid: unique identifier of the node of interest
        :param limit: maximum number of neighbours to return
        :return: db_id's of neighbours
        """

        q = "match (n {db_id: %d})-[r:INF]->(p) return p.db_id as db_id limit %d" % (dbid, limit)
        answer = self._driver.execute_query(q)
        return answer

    def breadth_first_search(self,
                             dbid: int,
                             limit: int,
                             max_depth: int):

        """
        Function that computes a breath first search in the neo4j graph

        :param dbid: db_id of the start node in the BFS
        :param limit: maximum no of neighbours to get for a node
        :param max_depth: maximum depth of the BFS

        :return: list of nodes found by BFS alongside their depth
        """

        depth = 0
        q = LifoQueue()
        result = list()

        q.put({'db_id': dbid, 'depth': depth})
        result.append({'db_id': dbid, 'depth': depth})

        while not q.empty() and depth < max_depth:
            depth += 1

            node = q.get()
            neighbours = self.get_neighbours(dbid=node['db_id'], limit=limit)

            for neighbour in neighbours:
                neighbour['depth'] = depth
                q.put(neighbour)
                result.append(neighbour)

        return result
