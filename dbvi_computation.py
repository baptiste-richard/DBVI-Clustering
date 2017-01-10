"""
Heuristic for high DBVI clustering
@authors: Baptiste R. & Vincent F. & Antoine C.
"""

import pandas as pd 
import numpy as np 
import matplotlib as plt 
plt.style.use('ggplot')
import networkx as nx
import scipy as sp
from scipy import spatial
import copy

d = 2 # feature space dimension

# DVBI is a dictionary where we will store all of our results
'''
[0] --> complete undirected MR Graph
[1] --> Minimum Spanning Tree (MST) coresponding to MR Graph
[2] --> Density Sparsness of Cluster (DSC), i.e. max weight of MST
[3] --> Reduced graph including internal nodes from previously created MST
[4] --> List of Lists correponding to internal nodes coordinates [0],[1] and core distances [2]
[5] --> List of smallest MR distances to other clusters
[6] --> Validity Index of cluster (end result)
'''

###############################################################################
# take dataframe as input and create list of clusters to compute DVBI
def create_clusters_list(dataframe):
    clusters = []
    cluster_list = dataframe['cluster'].unique()
    for cluster_name in cluster_list:
        cluster = dataframe[dataframe['cluster']==cluster_name]
        cluster_list = cluster[['x1', 'x2']].values.tolist()
        clusters.append(cluster_list)
    return(clusters)
###############################################################################


###############################################################################
# core distance definition
def core_distance(cluster,pos_clus):
    core_distance = 0
    active_cluster = copy.copy(cluster)
    x = active_cluster.pop(pos_clus)
    knn_sum = 0
    for i in range(0,len(active_cluster)):
        knn_sum += (1/sp.spatial.distance.euclidean(x,active_cluster[i][0:2]))**d
    core_distance = (knn_sum/len(active_cluster))**(float(-1)/float(d))
    return(core_distance)
###############################################################################


###############################################################################
# core distance definition
def compute_core_distances(clusters):
    for c in range(0,len(clusters)):
        for i in range(0,len(clusters[c])):
            clusters[c][i].append(core_distance(clusters[c],i))
###############################################################################


###############################################################################
# MR distance definition
def mr_distance(cluster,pos_clus_x,pos_clus_y):
    active_cluster = list(cluster)
    euc_dist_xy = sp.spatial.distance.euclidean(active_cluster[pos_clus_x][0:2],active_cluster[pos_clus_y][0:2])
    return max(cluster[pos_clus_x][2],cluster[pos_clus_y][2],euc_dist_xy)
###############################################################################


###############################################################################
# MR Graphs definition using networkx
# MR graph is undirected, complete - nodes correpond to data points in cluster, weights to MR distance
# Note: we can verify that we get n(n-1)/2 arcs for each complete graph
def create_mr_graph(clusters,DBVI):
    for k in range(0,len(clusters)):
        G = nx.Graph()
        for i in range(0,len(clusters[k])):
            G.add_node(i,pos=[clusters[k][i][0],clusters[k][i][1]])

        for j in range(0,len(clusters[k])):
            for i in range(1+j,len(clusters[k])):
                G.add_edge(j, i, weight = mr_distance(clusters[k],j,i))
        DBVI["cluster%s"%k] = [G]
###############################################################################

###############################################################################
# compute Density Sparsness of Cluster as the max weight of our MST
def DSC(DBVI):
    for k in range(0,len(DBVI)):
        # 1. minimum spanning tree
        min_tree = nx.minimum_spanning_tree(DBVI["cluster%s"%k][0])

        # 2. compute maximum weight of this MST
        max_weight = 0
        for j in range(0,len(min_tree.edges())):
            if min_tree.edges(data=True)[j][2]['weight'] > max_weight:
                max_weight = min_tree.edges(data=True)[j][2]['weight']
        
        # 3. store results in existing dictionary
        DBVI["cluster%s"%k].append(min_tree)
        DBVI["cluster%s"%k].append(max_weight)
###############################################################################

###############################################################################
# compute Density Separation Between Pairs of Clusters (DSPC)
def DSPC(DBVI,clusters):
    # 1. create subgraph of internal nodes only - position [3] of dictionnary
    for k in range(0,len(DBVI)):
        min_tree_int = DBVI["cluster%s"%k][1].copy()
        for node in min_tree_int.degree().keys():
            if min_tree_int.degree()[node] == 1:
                min_tree_int.remove_node(node)
        DBVI["cluster%s"%k].append(min_tree_int)
    
    # 2. transform graph object in list of list to compute distances easily - position [4] of dictionnary
        X = [] 
        for i in range(0,len(DBVI["cluster%s"%k][3].nodes(data=True))):
            X.append(DBVI["cluster%s"%k][3].nodes(data=True)[i][1]['pos'])
        DBVI["cluster%s"%k].append(X)
    
    # 3. find core-distance initially computed
        for p in range(0,len(DBVI["cluster%s"%k][4])):
            a = DBVI["cluster%s"%k][4][p][0]
            b = DBVI["cluster%s"%k][4][p][1]
            for i in range(0,len(clusters[k])):
                if a == clusters[k][i][0] and b == clusters[k][i][1]:
                    DBVI["cluster%s"%k][4][p].append(clusters[k][i][2])
            
    # 4. create variables in dictionary for smallest mr distances to other clusters
        smallest_mr_dist = [0] * len(DBVI)
        DBVI["cluster%s"%k].append(smallest_mr_dist)
            
    # 5. compute density separation for each cluster - iterate on each point of each clusters 2 by 2
    for k1 in range(0,len(DBVI)):
        for k2 in range(1+k1,len(DBVI)):
            dspc = 1000000
            for p1 in range(0,len(DBVI["cluster%s"%k1][4])):
                for p2 in range(0,len(DBVI["cluster%s"%k2][4])):
                    euc_dist = sp.spatial.distance.euclidean(DBVI["cluster%s"%k1][4][p1][0:2],DBVI["cluster%s"%k2][4][p2][0:2])
                    mr_dist = max(euc_dist,DBVI["cluster%s"%k1][4][p1][2],DBVI["cluster%s"%k2][4][p2][2])
                    if mr_dist < dspc:
                        dspc = mr_dist
            DBVI["cluster%s"%k1][5][k2] = dspc
            DBVI["cluster%s"%k2][5][k1] = dspc
###############################################################################


###############################################################################
# compute Validity Index of each cluster
def VIC(DBVI):
    for k in range(0,len(DBVI)):
        min_dspc = min(i for i in DBVI["cluster%s"%k][5] if i > 0)
        dsc = DBVI["cluster%s"%k][2]
        V = (min_dspc-dsc)/(max(min_dspc,dsc))
        DBVI["cluster%s"%k].append(V)
###############################################################################


###############################################################################
# compute DBVI
def DBVI_comp(all_data):
    DBVI = {}
    clusters = create_clusters_list(all_data)
    compute_core_distances(clusters)
    create_mr_graph(clusters,DBVI)
    DSC(DBVI)
    DSPC(DBVI,clusters)
    VIC(DBVI)
    DBVIndex = 0    
    print("--- DBVI Summary ---")
    print("")
    print("%s Data Points & %s Clusters"%(len(all_data),len(DBVI)))
    print("")
    for k in range(0,len(DBVI)):
        C = len(DBVI["cluster%s"%k][0].nodes())
        DBVIndex += C*DBVI["cluster%s"%k][6]/len(all_data)
        print("- Cluster %s: %s point - VIC = %s"%(k,C,DBVI["cluster%s"%k][6]))
    print("")
    print("--> DBVI = %s"%DBVIndex)
    return(DBVIndex,DBVI)
###############################################################################
