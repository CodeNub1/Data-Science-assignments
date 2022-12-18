import pandas as pd
import networkx as nx 
import matplotlib.pyplot as plt

# Load the dataset
G = pd.read_csv("C:/Users/Sidha/OneDrive/Desktop/Datasets/Network analytics/flight_hault.csv",header=None)
G.columns



#Giving column names
G.columns=["ID","Name","City","Country","IATA_FAA","ICAO","Latitude","Longitude","Altitude","Time","DST","Tz database time"]


G = G.iloc[0:50, 0:13]
G.info()


g = nx.DiGraph()

g = nx.from_pandas_edgelist(G, source = 'Country', target = 'IATA_FAA',create_using=nx.DiGraph)
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
