#! /bin/bash
export NEO4J_USER=neo4j
export NEO4J_PASS=tibi1932

if ! which "neo4j-admin.bat" >/dev/null || ! which "cypher-shell.bat" >/dev/null ; then
    echo "Cannot find neo4j binaries"
    echo "Please make sure that the neo4j binaries are in \$PATH"
    exit 1
fi

echo "Preparing to hydrate database"
read -p "Ensure neo4j is stopped and that any database files have been removed. Then press enter."
echo "Importing data"
neo4j-admin.bat import --id-type=INTEGER --multiline-fields=true --nodes dbinfo.csv --nodes conduit_pipe.csv --nodes actor_process.csv --nodes schema.csv --nodes paths.csv --nodes es_file.csv --nodes ctx_cadets_context.csv --nodes conduit_socket.csv --nodes cont_file.csv --nodes net.csv --nodes conduit_ptty.csv --nodes store_file.csv --relationships named.csv --relationships inf.csv
echo "Data import complete"
read -p "Now start neo4j, wait for it to come up, then press enter."
echo -n "Building indexes..."
cypher-shell.bat -u$NEO4J_USER -p$NEO4J_PASS >/dev/null <<EOF
CREATE INDEX ON :Node(db_id);
CREATE INDEX ON :Actor(uuid);
CREATE INDEX ON :Object(uuid);
CREATE INDEX ON :Store(uuid);
CREATE INDEX ON :EditSession(uuid);
CREATE INDEX ON :Conduit(uuid);
CREATE INDEX ON :StoreCont(uuid);
CREATE INDEX ON :Path(path);
CREATE INDEX ON :Net(addr);
CALL db.awaitIndexes();
EOF
echo "Done"
echo "Database hydrated"
