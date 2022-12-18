import pandas as pd
import networkx as nx 
import matplotlib.pyplot as plt

# Load the dataset
G = pd.read_csv("C:/Users/Sidha/OneDrive/Desktop/Datasets/Network analytics/connecting_routes.csv",header=None)
G.columns



#deleting column 6 and 7
G=G.drop(6,axis=1)
G=G.drop(7,axis=1)
G.shape
#Giving column names
G.columns=['flights','id','main airport','main airport id','destination','destination id','machinery']


G = G.iloc[0:1000, 0:8]
G.info()


g = nx.DiGraph()

g = nx.from_pandas_edgelist(G, source = 'main airport', target = 'destination',create_using=nx.DiGraph)
print(nx.is_directed(g))
print(nx.info(g))

#nodes
g.nodes()


#edges
g.edges()
len(g.edges())




#visualising with spring layout
pos = nx.spring_layout(g)
nx.draw_networkx(g, pos, node_size = 15, node_color = 'red')


#visualising with circular layout
pos = nx.circular_layout(g)
nx.draw_networkx(g, pos, node_size = 15, node_color = 'red')

#visualising with kamada_kawai_layout
pos = nx.kamada_kawai_layout(g)
nx.draw_networkx(g, pos, node_size = 15, node_color = 'red')

#visualising with spectral_layout
pos = nx.spectral_layout(g)
nx.draw_networkx(g, pos, node_size = 15, node_color = 'red')

#visualising with shell layout
pos = nx.shell_layout(g)
nx.draw_networkx(g, pos, node_size = 15, node_color = 'red')





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

