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
fc = hier.fcluster(Z, 3, criterion='distance')
cluster=[[]]
# print the clusters 
for k in range(1,max(fc)):
    print("Flat cluster",k,": ", list(np.where(fc==k)[0]))
    cluster.append(list(np.where(fc==k)[0]))

# make list from list of list cluster
flat_list = []
for sublist in cluster:
    for item in sublist:
        flat_list.append(item)
print("Flat cluster list: ", flat_list)
    
""" 
trying to generate adjac from linkage
"""

# Cutting the dendrogram with cut_tree
from scipy import cluster
Z = cluster.hierarchy.ward(Data)
cutree = cluster.hierarchy.cut_tree(Z,None,3)
AZ = cluster.hierarchy.ward(cutree)

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.cuts import conductance
from networkx.algorithms.cuts import cut_size
from networkx.algorithms.cuts import volume
from networkx.algorithms.community import quality as qu
# Create a networkx graph object
new_graph = nx.Graph() 

adjac = [(0,0)] # should not make any difference
n = len(Z)
for bb in Z:
    n=n+1
    x,y = (tuple(bb[:-2]))
    adjac.append(tuple([x,n]))
    adjac.append(tuple([y,n]))
    # print(adjac)
print(adjac)
print(Z)

new_graph.add_edges_from(adjac)
 
# Draw the resulting graph
nx.draw(new_graph, with_labels=True, font_weight='bold')

S = [6,7,13]
T = [4,15,8,14,3,11,1,10,0]


# Conductance
sumOfCond = []

for i in range(len(cluster)):
    sumOfCond.append(conductance(new_graph,cluster[i]))
    
condScoreS = conductance(new_graph,S)
condScoreT = conductance(new_graph,T)
overallCond = min(sumOfCond)

# Modularity Communities
barbMod = list(greedy_modularity_communities(new_graph))

# Modularity Score
barbModScore =  qu.modularity(new_graph, barbMod)

# edge_betweenness_centrality
barbedgeBetweenness=  nx.edge_betweenness_centrality(new_graph,None,False)
barbaverageEdge = sum(barbedgeBetweenness.values())/len(barbedgeBetweenness)
barbtotalEdge = sum(barbedgeBetweenness.values())

# print sets of nodes, one for each community.
print("Communities: ", barbMod)

# Modularity Score
print("Modularity: ", barbModScore)

# Conductance Score
print("Conductance for: ", S, condScoreS)
print("Conductance for: ", T, condScoreT)
print("Overall Conductance: ", overallCond)


# edge_betweenness_centrality
print("Edge Betweenness Centrality Score: ", barbaverageEdge)
