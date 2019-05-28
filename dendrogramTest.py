# Importing the libs
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd

# Importing test csv file with pd
# dataset = pd.read_csv('TestData.csv')
# Data = dataset.iloc[:, [1,2]].values

# Test-Data
Data = [[1, 2],
 [2, 2],
 [3, 4],
 [2, 3],
 [1, 1],
 [4, 5],
 [5, 6],
 [4, 7],
 [3, 2],
 [6, 3]]


# Creating a scatter plot
x,y = list(zip(*Data))
plt.scatter(x, y)
plt.show()


# Creating a dendrogram
from scipy.spatial.distance import pdist
import scipy.cluster.hierarchy as hier
dendrogram = hier.dendrogram(hier.linkage(Data, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
plt.show()


# Cutting the dendrogram
distanceMatrix = pdist(Data)
Z = hier.linkage(distanceMatrix, method='complete')

# cut dendrogram at certain distance
fc = hier.fcluster(Z, 2 , criterion='distance')

# print the clusters 
for k in range(1,max(fc)):
    print(np.where(fc == k))


import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.cuts import conductance
from networkx.algorithms.community import quality as qu

# Create a networkx graph object
my_graph = nx.Graph() 
 
# Add edges to to the graph object
# Each tuple represents an edge between two nodes
my_graph.add_edges_from([
                        (0,1), 
                        (1,2), 
                        (2,3), 
                        (2,4), 
                        (3,4)])
 
# Draw the resulting graph
nx.draw(my_graph, with_labels=True, font_weight='bold')

# Modularity Communities
c = list(greedy_modularity_communities(my_graph))

# print sets of nodes, one for each community.
print("Communities: ", c)

# Modularity Score
print("Modularity Score: ", qu.modularity(my_graph, c))

# edge_betweenness_centrality
print("Edge Betweenness Centrality: ", nx.edge_betweenness_centrality(my_graph))

# Conductance
# S (sequence) – A sequence of nodes in my_graph.
# T (sequence) – A sequence of nodes in my_graph.
S = [0,1]
T = [2,3,4]
print("Conductance: ", conductance(my_graph, S, T))
