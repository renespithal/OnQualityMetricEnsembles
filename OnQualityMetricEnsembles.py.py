import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist
import scipy.cluster.hierarchy as hier
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.cuts import conductance
from networkx.algorithms.cuts import volume
from networkx.algorithms.cuts import cut_size
from networkx.algorithms.community import quality as qu

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
# Creating Linkage Matrix Z
distanceMatrix = pdist(Data, 'euclidean')
Z = hier.linkage(distanceMatrix, 'ward')

# Creating a dendrogram
dendrogram = hier.dendrogram(Z)
plt.title('Intersect-Distance Intervals')
plt.xlabel('Index')
plt.ylabel('Distance')
plt.show()

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
    #originalAdjac.append(tuple([x,y]))
    #originalAdjac.append(tuple([y,x]))


original_graph.add_edges_from(originalAdjac)

nx.draw(original_graph, with_labels=True, font_weight='bold')


# score dicts
# key = cutDistance 
# value = Score
myModularityEnsemble = {}
myModY = []
myModScoreEnsemble = {}
modularityEnsemble = {}
modY = []
conductanceEnsemble = {}
condY = []
edgeEnsemble = {}
edgeY = []

# cut dendrogram at intersect-distance intervals where clusters were formed
cutIntervals = []
for xs in Z:
    # +0.009 to cut above cluster forming and not at exact point of cluster
    # forming - to get the actual clusters at that distance.
    cutIntervals.append(xs[2]+0.009)
    #print(cutIntervals)
rCutIntervals = np.around(cutIntervals, decimals=4)
intersectIntervals = sorted(set(rCutIntervals))

#intersect-distance intervals
cutDistance = intersectIntervals

# cut dendrogram at equi-distance intervals from 1 to max distance where clusters were formed
# [1,2,...,max]
#cutDistanceRange = list(range(int(round(max(intersectIntervals)))+1))

#equi-distance intervals 
#cutDistance = cutDistanceRange[1:]

#specific-cut distance
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


    # Create a networkx graph object
    new_graph = nx.Graph() 
    
    # adjac will be used to create networkx graph
    adjac = []
    # az will be used to create adjacency matrix
    az=[]
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
        ax=(int(bb[0]))
        ay=(int(bb[1]))
        nodesXy = (ax,ay)
        nodesYx = (ay,ax)
        nodesNx = (ax,n)
        nodesNy = (ay,n)
        az.append(nodesXy)
        az.append(nodesYx)
        az.append(nodesNx)
        az.append(nodesNy)
        
        adjac.append(tuple([x,n]))
        adjac.append(tuple([y,n]))
        #adjac.append(tuple([x,y]))
        #adjac.append(tuple([y,x]))
    
    new_graph.add_edges_from(adjac)
     
    # Draw the resulting graph
    plt.clf()
    nx.draw(new_graph, with_labels=True, font_weight='bold')

    
    
    # Creating adjac matrix
    adjac_matrix = np.zeros((len(Data), len(Data)),dtype=int)
  
    # adjac matrix #1
    # combine every element in each cluster with the rest of the elements
    # and update the adjacency matrix accordingly
    ajModList = []
    for xi in range(len(clusterL)):
        ajModList.append([])
        for xj in range(len(clusterL[xi])):
            for xk in range(len(clusterL[xi])):
                if clusterL[xi][xj] != clusterL[xi][xk]:
                    ajModList.append((clusterL[xi][xj], clusterL[xi][xk]))
                    adjac_matrix[clusterL[xi][xj]][clusterL[xi][xk]]=1
    
    # adjac matrix #2
    # combine every node like shown on graph
    adjac_matrix2 = np.zeros((len(Data)+len(Z), len(Data)+len(Z)),dtype=int)
    for nodex in az:
        adjac_matrix2[nodex] = 1
    
    
    G1 = nx.from_numpy_matrix(adjac_matrix)
    B1 = nx.modularity_matrix(G1)
    
    G2 = nx.from_numpy_matrix(adjac_matrix2)
    B2 = nx.modularity_matrix(G2)
    
    ajModScore = []
    for am in range(len(ajModList)):
        if ajModList[am] == []:
            continue
        ajModScore.append(ajModList[am])
    
           # Creating adjac matrix
    adjac_matrix3 = np.zeros((len(Data)+len(Z), len(Data)+len(Z)),dtype=int)
  
    # adjac matrix #1
    # combine every element in each cluster with the rest of the elements
    # and update the adjacency matrix accordingly
    component = list(nx.connected_components(G2))
    blabla = []
    for x in range(len(component)):
        blabla.append(list(component[x]))
        
    for xi in range(len(blabla)):
        for xj in range(len(blabla[xi])):
            for xk in range(len(blabla[xi])):
                #if blabla[xi][xj] != blabla[xi][xk]:
                    adjac_matrix3[blabla[xi][xj]][blabla[xi][xk]]=1
    
    myNewModScore = 0.0
    myB = B2.tolist()
    for x in range(len(myB)-1):
       for y in range(len(myB)-1):
           if adjac_matrix2[x][y] == 1:
               myNewModScore = myNewModScore + myB[x][y]
    if len(az) != 0:
         myNewModScore = myNewModScore / len(az)  
    myModularityEnsemble.update({cutDistance[a]: myNewModScore})
    myModY.append(myNewModScore)
    print("\n")
    print("adjacency matrix for cut size: ", cutDistance[a])
    print("\n")
    print("from clusterL")
    print(adjac_matrix)
    print("\n")
    print("from nodelist")
    print(adjac_matrix2)
    print("\n")
    # make list from list of list cluster
    flat_list = []
    for sublist in clusterL:
        for item in sublist:
            flat_list.append(item)
    #print("Flat cluster list: ", flat_list)
    #print("\n")


    
    # Modularity Communities
    Com = list(greedy_modularity_communities(new_graph))
    print(Com)
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
    G3 = nx.from_numpy_matrix(adjac_matrix3)
    myModScore = 0 
    for x in az:
        myModScore = myModScore + B2.item(x)
    if len(az) != 0:
        myModScore = myModScore / len(az)
    myModScoreEnsemble.update({cutDistance[a]: myModScore})
    print("Modularity: ", ModScore)
    print("\n")
    print("mynew modscore: ", myNewModScore)
    print("\n")
    print("My Modularity: ", myModScore)
    print("\n")

    modularityEnsemble.update({cutDistance[a]: ModScore})
    modY.append(ModScore)
    
    # Conductance Score
    sumOfCond = []
    
    for i in range(len(clusterL)):
        if clusterL[i] == []:
            # catch Division by zero
            continue
        if nx.volume(original_graph,clusterL[i]) == 0:
            continue
        # calculate Cond for current Cluster in list of Clusters 
        currentDist = cutDistance[a]
        cutCounter = 0
        for c in Z[:,2]:
            if currentDist < c:
                cutCounter = cutCounter + 1
        currentCond = cutCounter/nx.volume(original_graph,clusterL[i])
        # currentCond = conductance(original_graph,clusterL[i])
        sumOfCond.append(currentCond)
        print("Conductance for: ", clusterL[i], " = ", currentCond,"cut-size: ", cutCounter)
        
    overallCond = min(sumOfCond)
    
    print("Overall Conductance: ", overallCond)
    print("\n")
    conductanceEnsemble.update({cutDistance[a]: overallCond})
    condY.append(overallCond)
    
    # edge betweenness centrality Score
    print("Edge Betweenness Centrality Score: ", averageEdge)
    print("Edge Betweenness Centrality Score: ", totalEdge)
    edgeEnsemble.update({cutDistance[a]: totalEdge})
    edgeY.append(totalEdge)

bestModCut = max(modularityEnsemble, key=modularityEnsemble.get)
bestMyModCut = max(myModScoreEnsemble, key=myModScoreEnsemble.get)
bestMyNewModCut = max(myModularityEnsemble, key=myModularityEnsemble.get)
bestConduc = min(conductanceEnsemble, key=conductanceEnsemble.get)
bestEdgeCut = min(edgeEnsemble, key=edgeEnsemble.get)

print('best score for: \nNetworkX-Modularity: {} at Cut = {} \nTextbook-Modularity: {} at Cut = {}  \nConductance: {} at Cut = {} \nEdge Betweenness Centrality:{} at Cut = {}'.format(modularityEnsemble[bestModCut],bestModCut,myModularityEnsemble[bestMyNewModCut],bestMyNewModCut, conductanceEnsemble[bestConduc],bestConduc,edgeEnsemble[bestEdgeCut],bestEdgeCut))

#print('best score for: \nModularity: {} at Cut = {} \nMyModularity: {} at Cut = {} \nMyNewModularity: {} at Cut = {}  \nConductance: {} at Cut = {} \nEdge Betweenness Centrality:{} at Cut = {}'.format(modularityEnsemble[bestModCut],bestModCut,myModScoreEnsemble[bestMyModCut],bestMyModCut,myModularityEnsemble[bestMyNewModCut],bestMyNewModCut, conductanceEnsemble[bestConduc],bestConduc,edgeEnsemble[bestEdgeCut],bestEdgeCut))


#PLOTS SCORES

#Modularity
# x axis values 
xMod = cutDistance
# corresponding y axis values 
yMod = modY
# plotting the points  
plt.plot(xMod, yMod)
plt.title('Modularity - Intersect-Distances - Networkx Implementation')
# naming the x axis 
plt.xlabel('cut Distance') 
# naming the y axis 
plt.ylabel('Modularity Score') 

# x axis values 
xMyMod = cutDistance
# corresponding y axis values 
yMyMod = myModY
# plotting the points  
plt.plot(xMyMod, yMyMod)
plt.title('Modularity - Intersect-Distances - Textbook Implementation')
# naming the x axis 
plt.xlabel('cut Distance') 
# naming the y axis 
plt.ylabel('Modularity Score') 

#Conductance
# x axis values 
xCond = cutDistance
# corresponding y axis values 
yCond = condY
# plotting the points  
plt.plot(xCond, yCond)
plt.title('Conductance - Intersect-Distances - Networkx Implementation')
# naming the x axis 
plt.xlabel('cut Distance') 
# naming the y axis 
plt.ylabel('Conductance Score') 

#Edge Betweenness Centrality
# x axis values 
xEdge = cutDistance
# corresponding y axis values 
yEdge = edgeY
# plotting the points  
plt.plot(xEdge, yEdge)
plt.title('Edge Betweenness Centrality - Intersect-Distances - Networkx Implementation')
# naming the x axis 
plt.xlabel('cut Distance') 
# naming the y axis 
plt.ylabel('Edge Betweenness Centrality Score') 
