from neo4j.v1 import GraphDatabase


class Neo4JDriver(object):

    def __init__(self,
                 url: str,
                 user: str,
                 pswd: str):
        self._driver = GraphDatabase.driver(url, auth=(user, pswd), encrypted=False)

    def execute_query(self,
                      query: str,
                      **kwargs):
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
