import networkx as nx
dfg = nx.DiGraph()
for i in range(10):
    dfg.add_node(i)

dfg.add_edge(0, 1)
dfg.add_edge(0, 4)
dfg.add_edge(0, 7)
