from data_loader import load_local_data
import networkx as nx
from utils import graph_colors
import matplotlib.pyplot as plt
from pscn import ReceptiveFieldMaker

mutag_dataset = load_local_data('./data', 'mutag')
X, y = zip(*mutag_dataset)

nx_g = X[0].nx_graph

plt.figure(figsize=(5, 10))
pos = nx.layout.kamada_kawai_layout(nx_g)
plt.subplot(2, 1, 1)
nx.draw(nx_g
        , pos=pos
        , with_labels=True
        , labels=nx.get_node_attributes(nx_g, 'attr_name')  # the features of the nodes are named 'attr_name' by default
        , node_color=graph_colors(nx_g))
plt.title('A Mutag molecule example with the features of the nodes')
plt.subplot(2, 1, 2)
nx.draw(nx_g
        , pos=pos
        , with_labels=True
        , node_color=graph_colors(nx_g))
plt.title('A Mutag molecule example. Nodes are named w.r.t the construction in data_loader')
plt.show()

rf_maker = ReceptiveFieldMaker(nx_g, w=10, k=5)

nx_normalized = rf_maker.normalize_graph(nx_g, vertex=15)

pos = nx.layout.kamada_kawai_layout(nx_normalized)
nx.draw(nx_normalized
        , pos=pos
        , with_labels=True
        , labels=nx.get_node_attributes(nx_normalized, 'attr_name')
        , node_color=graph_colors(nx_normalized))
plt.title('Normalize receptive field using node 15 as root node')
plt.show()

nx_normalized.nodes(data=True)

nx_relabel = nx.relabel_nodes(nx_normalized, nx.get_node_attributes(nx_normalized, 'labeling'))

print([x[1] for x in sorted(nx.get_node_attributes(nx_relabel, 'attr_name').items(), key=lambda x: x[0])])

forcnn = rf_maker.make_()

plt.figure(figsize=(10, 5))
for i in range(len(rf_maker.all_subgraph)):
    g = rf_maker.all_subgraph[i]
    pos = nx.layout.kamada_kawai_layout(nx_g)
    plt.subplot(2, 5, i + 1)
    nx.draw(g
            , pos=pos
            , with_labels=True
            , labels=nx.get_node_attributes(g, 'attr_name')
            , node_color=graph_colors(g))
plt.suptitle('All the receptive fields used in the CNN')
plt.show()

rf_maker = ReceptiveFieldMaker(nx_g, w=10, k=5,
                               one_hot=7)  # one_hot is the number of different attributes when they are discrete

print(rf_maker.make_())
