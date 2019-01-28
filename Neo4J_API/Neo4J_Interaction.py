from Neo4J_API.Neo4J_Driver import Neo4JDriver


class Neo4JInteraction(object):

    def __init__(self,
                 driver: Neo4JDriver):
        self._driver = driver

    def get_all_files(self):
        q = "match (n:Store) return n.uuid, n.db_id, n.ty, n.meta_hist " \
            "LIMIT 1"
        answer = self._driver.execute_query(q)
        return answer

    def get_all_processes(self):
        q = "match (n:Actor) return n.ty, n.rgid, n.pid, n.ruid, n.suid, n.cmdline, n.login_name " \
            "LIMIT 1"
        answer = self._driver.execute_query(q)
        return answer

    def get_node_degree(self,
                        ty='process'):
        q = "match (n) " \
            "with n, size(()-[:INF]->(n)) as degree " \
            "where n.ty = '%s' " \
            "return n.uuid, degree " \
            "LIMIT 100"

        answer = self._driver.execute_query(q)
        return answer


neo4Jdriver = Neo4JDriver(
    url='bolt://localhost:7687',
    user='neo4j',
    pswd='opus'
)

my_object = Neo4JInteraction(neo4Jdriver)

print(my_object.get_all_files())
print(my_object.get_all_processes())
print(my_object.get_node_degree())
