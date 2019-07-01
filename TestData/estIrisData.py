# Code source: GaÃ«l Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
#from sklearn.decomposition import PCA

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# Creating a dendrogram
from scipy.spatial.distance import pdist
import scipy.cluster.hierarchy as hier
import numpy as np

dendrogram = hier.dendrogram(hier.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
plt.show()

""" 
trying to generate adjac from linkage

"""




# Cutting the dendrogram with cut_tree
from scipy import cluster
Z = cluster.hierarchy.ward(X)
cutree = cluster.hierarchy.cut_tree(Z,None,3)
AZ = cluster.hierarchy.ward(cutree)


dendrogram = hier.dendrogram(hier.linkage(AZ, method = 'ward'))
plt.title('Dendrogram2')
plt.xlabel('Index')
plt.ylabel('Distance')
plt.show()

import networkx as nx
# Create a networkx graph object
new_graph = nx.Graph() 

adjac = [(0,0)]
for bb in AZ:
    adjac.append((tuple(bb[:-2])))
    print(adjac)


new_graph.add_edges_from(adjac)
 
# Draw the resulting graph
nx.draw(new_graph, with_labels=True, font_weight='bold')


"""

END

"""


# Cutting the dendrogram fcluster
distanceMatrix = pdist(X)
Z = hier.linkage(distanceMatrix, method='complete')

# cut dendrogram at certain distance
fc = hier.fcluster(Z, 3 , criterion='distance')

# print the clusters 
for k in range(1,max(fc)):
    print(np.where(fc == k))
