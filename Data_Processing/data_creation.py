import numpy


def create_data():
    graph_labels = open('graph_labels.txt', 'a')
    graph_labels.truncate(0)
    for interator in range(0, 100):
        print(1, file=graph_labels)
    for iterator in range(0, 100):
        print(-1, file=graph_labels)

    graph_indicator = open('graph_indicator.txt', 'a')
    graph_indicator.truncate(0)
    for iterator in range(1, 201):
        print(iterator, file=graph_indicator)
        print(iterator, file=graph_indicator)

    graph_edges = open('graph_edges.txt', 'a')
    graph_edges.truncate(0)
    for iterator in range(1, 201):
        print(str(2 * iterator - 1) + ', ' + str(2 * iterator), file=graph_edges)

    node_attributes = open('node_attributes.txt', 'a')
    node_attributes.truncate(0)
    for iterator in range(0, 400):
        if iterator % 2 == 0 and iterator < 200:
            print('1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0',
                  file=node_attributes)
        elif iterator % 2 == 0:
            print('1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0',
                  file=node_attributes)
        elif iterator < 200:
            print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0',
                  file=node_attributes)
        else:
            print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0',
                  file=node_attributes)


print(2)
