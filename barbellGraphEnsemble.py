# -*- coding: utf-8 -*-
# Ensemble on barbell graph
"""
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.cuts import conductance
from networkx.algorithms.community import quality as qu


# barbell_graph

barb = nx.barbell_graph(4,0)
nx.draw(barb, with_labels=True, font_weight='bold')
S = [0,1,2,3]
T = [4,5,6,7]
cluster=[[0,1,2,3],[4,5,6,7]]

# Conductance
sumOfCond = []

for i in range(len(cluster)):
    sumOfCond.append(conductance(barb,cluster[i]))
    
condScoreS = conductance(barb,S)
condScoreT = conductance(barb,T)
overallCond = min(sumOfCond)

# Modularity Communities
barbMod = list(greedy_modularity_communities(barb))

# Modularity Score
barbModScore =  qu.modularity(barb, barbMod)

# edge_betweenness_centrality
barbedgeBetweenness=  nx.edge_betweenness_centrality(barb,None,False)
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
print("Edge Betweenness Centrality: ", barbedgeBetweenness)
