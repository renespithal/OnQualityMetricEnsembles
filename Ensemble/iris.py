import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.cuts import conductance
from networkx.algorithms.cuts import volume
from networkx.algorithms.cuts import cut_size
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

#nx.draw(original_graph, with_labels=True, font_weight='bold')


# score dicts
# key = cutDistance 
# value = Score
modularityEnsemble = {}
conductanceEnsemble = {}
edgeEnsemble = {}

# cut dendrogram at distances where clusters were formed
cutIntervals = []
for xs in Z:
    cutIntervals.append(xs[2])
rCutIntervals = np.around(cutIntervals, decimals=1)
intersectIntervals = sorted(set(rCutIntervals))
# cutDistance = intersectIntervals

# cut dendrogram at equi distances from 0 to max distance where clusters were formed
# [0,1,2,...,max]
cutDistanceRange = list(range(int(round(max(intersectIntervals)))+1))
cutDistance = cutDistanceRange

#cutDistance = [1]


# Calculate score and draw graph for each cut distance
for a in range (len(cutDistance)):
    # For Cut Distance 0 => show singleton clusters otherwise skip
    if cutDistance[a] == 0 and len(cutDistance) > 1:
        continue
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
    
    adjac = []
    n = len(Z)
    # make adjacency matrix from linkage matrix
    for bb in Z:
        # connection to the next node/cluster
        n=n+1
        if bb[2] == 0:
            continue
        # bb[2] > cutDistance[a] = kante fällt durch den cut weg / noise wird gezeichnet
        # x1 oder y1 < len(Z) = wird gezeichnet falls original data point => singleton cluster
        if bb[2] > cutDistance[a]:
            x1 = bb[0]
            y1 = bb[1]
            if x1 <= len(Z):              
                adjac.append(tuple([x1,x1]))
            if y1 <= len(Z):
                adjac.append(tuple([y1,y1]))
            continue
        x,y = (tuple(bb[:-2]))
        adjac.append(tuple([x,n]))
        adjac.append(tuple([y,n]))
    
    new_graph.add_edges_from(adjac)
     
    # Draw the resulting graph
    plt.clf()
    nx.draw(new_graph, with_labels=True, font_weight='bold')
    
    # Modularity Communities
    Com = list(greedy_modularity_communities(new_graph))
    
    # Modularity Score
    ModScore =  qu.modularity(new_graph, Com)
    
    # edge_betweenness_centrality
    edgeBetweenness=  nx.edge_betweenness_centrality(new_graph,None,False)
    averageEdge = sum(edgeBetweenness.values())/len(edgeBetweenness)
    totalEdge = sum(edgeBetweenness.values())
    
    print("\n")
    print("\n")
    print("Scores for Cut Distance: ", cutDistance[a])
    print("\n")
    
    # Modularity Score
    print("Modularity: ", ModScore)
    print("\n")
    modularityEnsemble.update({cutDistance[a]: ModScore})
    
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
    conductanceEnsemble.update({cutDistance[a]: overallCond})
    
    # edge betweenness centrality Score
    print("Edge Betweenness Centrality Score: ", averageEdge)
    print("Edge Betweenness Centrality Score: ", totalEdge)
    edgeEnsemble.update({cutDistance[a]: totalEdge})

bestModCut = max(modularityEnsemble, key=modularityEnsemble.get)
bestConduc = min(conductanceEnsemble, key=conductanceEnsemble.get)
bestEdgeCut = min(edgeEnsemble, key=edgeEnsemble.get)

print('best score for: \nModularity: {} at Cut = {} \nConductance: {} at Cut = {} \nEdge Betweenness Centrality:{} at Cut = {}'.format(modularityEnsemble[bestModCut],bestModCut,conductanceEnsemble[bestConduc],bestConduc,edgeEnsemble[bestEdgeCut],bestEdgeCut))
