import numpy
from random import randint


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
        for i in range(0, 10):
            print(iterator, file=graph_indicator)

    graph_edges = open('graph_edges.txt', 'a')
    graph_edges.truncate(0)
    for iterator in range(0, 100):
        base = 10 * iterator
        print(str(base + 1) + ', ' + str(base + 4), file=graph_edges)
        print(str(base + 2) + ', ' + str(base + 4), file=graph_edges)
        print(str(base + 3) + ', ' + str(base + 4), file=graph_edges)
        print(str(base + 4) + ', ' + str(base + 5), file=graph_edges)
        print(str(base + 6) + ', ' + str(base + 5), file=graph_edges)
        print(str(base + 7) + ', ' + str(base + 5), file=graph_edges)
        print(str(base + 8) + ', ' + str(base + 5), file=graph_edges)
        print(str(base + 5) + ', ' + str(base + 10), file=graph_edges)
        print(str(base + 9) + ', ' + str(base + 10), file=graph_edges)
    for iterator in range(100, 200):
        base = 10 * iterator
        print(str(base + 1) + ', ' + str(base + 8), file=graph_edges)
        print(str(base + 2) + ', ' + str(base + 8), file=graph_edges)
        print(str(base + 3) + ', ' + str(base + 8), file=graph_edges)
        print(str(base + 4) + ', ' + str(base + 8), file=graph_edges)
        print(str(base + 5) + ', ' + str(base + 8), file=graph_edges)
        print(str(base + 6) + ', ' + str(base + 8), file=graph_edges)
        print(str(base + 7) + ', ' + str(base + 8), file=graph_edges)
        print(str(base + 8) + ', ' + str(base + 9), file=graph_edges)
        print(str(base + 9) + ', ' + str(base + 10), file=graph_edges)

    node_attributes = open('node_attributes.txt', 'a')
    node_attributes.truncate(0)
    for iterator in range(0, 100):
        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0',
              file=node_attributes)
        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0',
              file=node_attributes)
        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0',
              file=node_attributes)
        print('1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0',
              file=node_attributes)
        print('1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0',
              file=node_attributes)
        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0',
              file=node_attributes)
        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0',
              file=node_attributes)
        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0',
              file=node_attributes)
        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0',
              file=node_attributes)
        print('1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0',
              file=node_attributes)
    for iterator in range(100, 200):
        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0',
              file=node_attributes)
        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0',
              file=node_attributes)
        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0',
              file=node_attributes)
        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0',
              file=node_attributes)
        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0',
              file=node_attributes)
        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0',
              file=node_attributes)
        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0',
              file=node_attributes)
        print('1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0',
              file=node_attributes)
        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0',
              file=node_attributes)
        print('1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0',
              file=node_attributes)


def create_basic_ssh_vs_sleep():
    graph_labels = open('graph_labels.txt', 'a')
    graph_labels.truncate(0)
    for interator in range(0, 500):
        print(1, file=graph_labels)
    for iterator in range(0, 500):
        print(-1, file=graph_labels)

    graph_indicator = open('graph_indicator.txt', 'a')
    graph_indicator.truncate(0)
    for iterator in range(1, 1001):
        for i in range(0, 2):
            print(iterator, file=graph_indicator)

    graph_edges = open('graph_edges.txt', 'a')
    graph_edges.truncate(0)
    for iterator in range(1, 2000, 2):
        print(str(iterator) + ', ' + str(iterator + 1), file=graph_edges)

    node_attributes = open('node_attributes.txt', 'a')
    node_attributes.truncate(0)
    for iterator in range(0, 500):
        indicator = randint(1, 10)
        print('1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ', file=node_attributes,
              end='')
        if indicator <= 7:
            print('0, 0, 1, 0', file=node_attributes)
        else:
            print('0, 0, 0, 1', file=node_attributes)

        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, ', file=node_attributes, end='')
        if indicator <= 7:
            print('0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0', file=node_attributes)
        elif indicator <= 9:
            print('1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0', file=node_attributes)
        else:
            print('0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0', file=node_attributes)

    for iterator in range(500, 1000):
        indicator = randint(1, 10)
        print('1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ', file=node_attributes,
              end='')
        if indicator <= 8:
            print('0, 0, 0, 1', file=node_attributes)
        else:
            print('0, 1, 0, 0', file=node_attributes)

        print('0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ', file=node_attributes, end='')
        if indicator <= 7:
            print('0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0', file=node_attributes)
        elif indicator <= 9:
            print('1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0', file=node_attributes)
        else:
            print('0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0', file=node_attributes)


def create_basic_4_class_comparison():
    graph_labels = open('graph_labels.txt', 'a')
    graph_labels.truncate(0)
    for interator in range(0, 500):
        print(1, file=graph_labels)
    for iterator in range(0, 500):
        print(2, file=graph_labels)
    for iterator in range(0, 500):
        print(3, file=graph_labels)
    for iterator in range(0, 500):
        print(4, file=graph_labels)

    graph_indicator = open('graph_indicator.txt', 'a')
    graph_indicator.truncate(0)
    for iterator in range(1, 2001):
        for i in range(0, 2):
            print(iterator, file=graph_indicator)

    graph_edges = open('graph_edges.txt', 'a')
    graph_edges.truncate(0)
    for iterator in range(1, 4000, 2):
        print(str(iterator) + ', ' + str(iterator + 1), file=graph_edges)

    node_attributes = open('node_attributes.txt', 'a')
    node_attributes.truncate(0)
    for iterator in range(0, 500):
        indicator = randint(1, 10)
        print('1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ', file=node_attributes,
              end='')
        if indicator <= 7:
            print('0, 0, 1, 0', file=node_attributes)
        else:
            print('0, 0, 0, 1', file=node_attributes)

        print('0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, ', file=node_attributes, end='')
        if indicator <= 7:
            print('0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0', file=node_attributes)
        elif indicator <= 9:
            print('1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0', file=node_attributes)
        else:
            print('0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0', file=node_attributes)

    for iterator in range(500, 1000):
        indicator = randint(1, 10)
        print('1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ', file=node_attributes,
              end='')
        if indicator <= 8:
            print('0, 0, 0, 1', file=node_attributes)
        else:
            print('0, 1, 0, 0', file=node_attributes)

        print('0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ', file=node_attributes, end='')
        if indicator <= 7:
            print('0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0', file=node_attributes)
        elif indicator <= 9:
            print('1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0', file=node_attributes)
        else:
            print('0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0', file=node_attributes)

    for iterator in range(500, 1000):
        indicator = randint(1, 10)
        print('1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ', file=node_attributes,
              end='')
        if indicator <= 8:
            print('1, 0, 0, 0', file=node_attributes)
        else:
            print('1, 0, 0, 0', file=node_attributes)

        print('0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, ', file=node_attributes, end='')
        if indicator <= 7:
            print('0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0', file=node_attributes)
        elif indicator <= 9:
            print('0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0', file=node_attributes)
        else:
            print('0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0', file=node_attributes)

    for iterator in range(500, 1000):
        indicator = randint(1, 10)
        print('1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ', file=node_attributes,
              end='')
        if indicator <= 8:
            print('0, 0, 0, 1', file=node_attributes)
        else:
            print('0, 1, 0, 0', file=node_attributes)

        print('0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, ', file=node_attributes, end='')
        if indicator <= 7:
            print('0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0', file=node_attributes)
        elif indicator <= 9:
            print('0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0', file=node_attributes)
        else:
            print('0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0', file=node_attributes)


create_basic_ssh_vs_sleep()
# create_basic_4_class_comparison()