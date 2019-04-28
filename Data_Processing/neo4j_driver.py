from neo4j.v1 import GraphDatabase


class Neo4JDriver(object):

    def __init__(self,
                 url: str,
                 user: str,
                 pswd: str):
        self._driver = GraphDatabase.driver(url, auth=(user, pswd), encrypted=False)

    def execute_query(self,
                      query: str):
        session = self._driver.session()
        query_answer = session.run(query)
        records_answer = query_answer.records()
        query_answer = list()
        for record in records_answer:
            query_answer.append(dict(record.items()))

        session.close()

        return query_answer

    def close_connection(self):
        self._driver.close()


class Neo4JInteraction(object):

    def __init__(self,
                 driver: Neo4JDriver):
        self._driver = driver

    def get_nodes(self, node_type=None):
        if node_type is None:
            q = "match (n) return n.db_id as db_id"
        else:
            q = "match (n: %s) return n.db_id as db_id, n.cmdline as cmd_line" % node_type
        answer = self._driver.execute_query(q)
        return answer

    def get_number_of_nodes(self,
                            node_type=None):
        if node_type is None:
            q = "match (n) return count(*)"
        else:
            q = "match (n: %s) return count(*)" % node_type
        answer = self._driver.execute_query(q)
        return answer

    def get_node_indegree(self,
                          uuid: str):
        q = "match p = ()-[r:INF]->(n {uuid: '%s'}) return count(*) as files_in_touch" % uuid
        answer = self._driver.execute_query(q)
        return answer

    def get_edges(self,
                  rel_type: str,
                  node_type1: str,
                  node_type2: str):
        q = "match p = (parent: %s)-[r: %s]->(children: %s) " \
            "return parent.db_id, children.db_id" % (
                node_type1, rel_type, node_type2)
        answer = self._driver.execute_query(q)
        return answer

    def get_number_of_edges(self,
                            rel_type: str,
                            node_type1: str,
                            node_type2: str):
        q = "match p = (parent: %s)-[r: %s]->(children: %s) " \
            "return count(*)" % (
                node_type1, rel_type, node_type2)
        answer = self._driver.execute_query(q)
        return answer

    def get_edge_attributes(self,
                            uuid: str):
        q = "match (n {uuid: '%s'}) return " \
            "n.euid as euid, " \
            "n.rgid as rgid, " \
            "n.pid as pid, " \
            "n.suid as suid, " \
            "n.egid as egid, " \
            "n.sgid as sgid, " \
            "n.db_id as db_id, " \
            "n.cmd_line as cmd_line, " \
            "n.login_name as login_name" % uuid

        answer = self._driver.execute_query(q)
        return answer

    def get_process_attributes(self,
                               uuid: str):
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

        q = "match (n:StoreCont {uuid: %s}) return n"
        answer = self._driver.execute_query(q)
        return answer
