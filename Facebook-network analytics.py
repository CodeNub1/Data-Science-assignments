import pandas as pd
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt

# Load the dataset
G = pd.read_csv("C:/Users/Sidha/OneDrive/Desktop/Datasets/Network analytics/facebook.csv")

G.shape


#renaming columns
G.columns=[0,1,2,3,4,5,6,7,8]

#creating adjacency matrix
a= nx.Graph()
g = nx.from_pandas_adjacency(G)





#visualising with circular layout
pos = nx.circular_layout(g)
nx.draw_networkx(g, pos, node_size = 15, node_color = 'red')

print(nx.is_directed(g))
print(nx.info(g))

#nodes
g.nodes()


#edges
g.edges()
len(g.edges())





# Degree Centrality
d = nx.degree_centrality(g)
print(d) 

# closeness centrality
closeness = nx.closeness_centrality(g)
print(closeness)

## Betweeness Centrality 
b = nx.betweenness_centrality(g) # Betweeness_Centrality
print(b)

## Eigen-Vector Centrality
evg = nx.eigenvector_centrality(g) # Eigen vector centrality
print(evg)

# cluster coefficient
cluster_coeff = nx.clustering(g)
print(cluster_coeff)

# Average clustering
cc = nx.average_clustering(g) 
print(cc)

