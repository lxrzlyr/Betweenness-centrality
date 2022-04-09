import networkx as nx
import random
import numpy as np
import timeit
start=timeit.default_timer()
G=nx.random_graphs.erdos_renyi_graph(30000,0.2, directed=False)
#for edge in G.edges():
#    G.edges[edge[0],edge[1]]['weight'] =random.randint(1, 5)
A=np.array(nx.adjacency_matrix(G).todense())
np.savetxt('matrix.txt',A,newline='\n', fmt='%u')
list = nx.algorithms.centrality.betweenness_centrality(G).values()
data = open("OUTPUT.txt", "w+")
for value in list:
    print(value, file=data)
end=timeit.default_timer()
print('The time is ', end-start)
print('The time is ', end-start, file=data)