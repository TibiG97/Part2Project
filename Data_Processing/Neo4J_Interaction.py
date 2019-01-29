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

    def get_nodes(self, node_type):
        q = "match (n: %s) return n.uuid" % node_type
        answer = self._driver.execute_query(q)
        return answer

    def get_edges(self,
                  rel_type: str,
                  node_type1: str,
                  node_type2: str):
        q = "match p = (parent: %s)-[r: %s]->(children: %s) " \
            "return parent.uuid, children.uuid" % (
                node_type1, rel_type, node_type2)
        answer = self._driver.execute_query(q)
        return answer

    def get_node_attributes(self, uuid: str):
        return 0

    def get_node_degree(self,
                        ty='process'):
        q = "match (n) " \
            "with n, size(()-[:INF]->(n)) as degree " \
            "where n.ty = '%s' " \
            "return n.uuid, degree " \
            "LIMIT 100" % ty

        answer = self._driver.execute_query(q)
        return answer


neo4Jdriver = Neo4JDriver(
    url='bolt://localhost:7687',
    user='neo4j',
    pswd='opus'
)

my_object = Neo4JInteraction(neo4Jdriver)

print(my_object.get_edges('INF', 'Store', 'Actor').__sizeof__())
print(my_object.get_edges('INF', 'Store', 'Store').__sizeof__())
print(my_object.get_edges('INF', 'Actor', 'Store').__sizeof__())
print(my_object.get_edges('INF', 'Actor', 'Actor').__sizeof__())
