import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.cuts import conductance
from networkx.algorithms.cuts import volume
from networkx.algorithms.community import quality as qu

# import Iris data from sklearn
iris = datasets.load_iris()
# we only take the first two features.
X = iris.data[:, :2]  
x = X[:, 0]
y = X[:, 1]
plt.scatter(x, y)
plt.show()
Data = X[:, :2]


# Creating a dendrogram
from scipy.spatial.distance import pdist
import scipy.cluster.hierarchy as hier
dendrogram = hier.dendrogram(hier.linkage(Data, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
plt.show()


# Creating Linkage Matrix Z
distanceMatrix = pdist(Data)
Z = hier.linkage(distanceMatrix, method='ward')


# Create a networkx graph before any cuts
original_graph = nx.Graph() 
originalAdjac = []
n = len(Z)
# make adjacency matrix from linkage matrix
for bb in Z:
    # connection to the next node/cluster
    # n = cluster ID
    n=n+1
    if bb[2] == 0:
        continue
    x,y = (tuple(bb[:-2]))
    originalAdjac.append(tuple([x,n]))
    originalAdjac.append(tuple([y,n]))

original_graph.add_edges_from(originalAdjac)


# cut dendrogram at distances where clusters were formed
cutIntervals = []
for xs in Z:
    cutIntervals.append(xs[2])
rCutIntervals = np.around(cutIntervals, decimals=1)
intersectIntervals = sorted(set(rCutIntervals))


# cut dendrogram at equi distances from 0 to max distance where clusters were formed
## cutDistance = list(range(int(round(max(intersectIntervals)))+1))
cutDistance = [3]


# Calculate score and draw graph for each cut distance
for a in range (len(cutDistance)):
    fc = hier.fcluster(Z, cutDistance[a], criterion='distance')
    clusterL=[]
    print("\n")
    print("Ensemble Scores for Cutsize: ", cutDistance[a])
    print("\n")
    print("Number of Clusters: ", max(fc))
    print("\n")
    # print the clusters 
    for k in range(1,max(fc)+1):
        print("Cluster",k,": ", list(np.where(fc==k)[0]))
        clusterL.append(list(np.where(fc==k)[0]))
    
    # make list from list of list cluster
    flat_list = []
    for sublist in clusterL:
        for item in sublist:
            flat_list.append(item)
    #print("Flat cluster list: ", flat_list)
    #print("\n")
    # Create a networkx graph object
    new_graph = nx.Graph() 
    
    adjac = [] # should not make any difference
    n = len(Z)
    # make adjacency matrix from linkage matrix
    for bb in Z:
        # connection to the next node/cluster
        n=n+1
        if bb[2] == 0:
            continue
        x,y = (tuple(bb[:-2]))
        adjac.append(tuple([x,n]))
        adjac.append(tuple([y,n]))
    
    new_graph.add_edges_from(adjac)
     
    # Draw the resulting graph
    nx.draw(new_graph, with_labels=True, font_weight='bold')
    
    # Modularity Communities
    barbMod = list(greedy_modularity_communities(new_graph))
    
    # Modularity Score
    barbModScore =  qu.modularity(new_graph, barbMod)
    
    # edge_betweenness_centrality
    barbedgeBetweenness=  nx.edge_betweenness_centrality(new_graph,None,False)
    barbaverageEdge = sum(barbedgeBetweenness.values())/len(barbedgeBetweenness)
    barbtotalEdge = sum(barbedgeBetweenness.values())
    print("\n")
    print("\n")
    print("Scores for Cut Distance: ", cutDistance[a])
    print("\n")
    # print sets of nodes, one for each community.
    # print("Communities: ", barbMod)
    # print("\n")
    # Modularity Score
    print("Modularity: ", barbModScore)
    print("\n")
    # Conductance Score
    sumOfCond = []
    
    for i in range(len(clusterL)):
        if clusterL[i] == []:
            # catch Division by zero
            continue
        if nx.volume(original_graph,clusterL[i]) == 0:
            continue
        # calculate Cond for current Cluster in list of Clusters 
        currentCond = conductance(original_graph,clusterL[i])
        sumOfCond.append(currentCond)
        print("Conductance for: ", clusterL[i], " = ", currentCond)
    overallCond = min(sumOfCond)
    print("Overall Conductance: ", overallCond)
    print("\n")
    
    # edge_betweenness_centrality
    print("Edge Betweenness Centrality Score: ", barbaverageEdge)
