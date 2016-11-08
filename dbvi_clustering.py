# DBVI Clustering
# 1. Kmeans ++ clustering implementation

import pandas as pd 
import numpy as np 
%matplotlib inline 
import matplotlib as plt 
plt.style.use('ggplot')
import random
from sklearn.cluster import KMeans
import networkx as nx

# using polar coordinates, we define a point generator on a ball with given radius and center
# radius is defined as a positive number

#problem dimension - set to R2 feature space
d = 2

def rand_ball_gen(radius,center):
    phi = random.uniform(0,2*np.pi)
    r = random.uniform(0,radius)
    
    x = r*np.cos(phi) + center[0]
    y = r*np.sin(phi) + center[1]

    return np.array([x,y])
    
    
# generate 4 distinct centers and generate dataset using those centers
c1 = np.array([0,0])
c2 = np.array([10,10])
c3 = np.array([-5,-5])
c4 = np.array([-5,5])

A = [rand_ball_gen(10,c1) for i in xrange(100)]
B = [rand_ball_gen(10,c2) for i in xrange(100)]
C = [rand_ball_gen(10,c3) for i in xrange(100)]
D = [rand_ball_gen(10,c4) for i in xrange(100)]
X = A + B + C + D
random.shuffle(X)


# Kmeans ++ algorithm on that data for k = 4 - kmeans ++ initialization

total_iteration = []
inertia = 0

for k in range(1,300):
    kmeans = KMeans(n_clusters = 4, init = 'k-means++', n_init = 1, max_iter = k, random_state = 92).fit(X)
    if inertia == kmeans.inertia_:
        num_iteration = k
        break
    inertia = kmeans.inertia_
total_iteration.append(num_iteration)

print("kmeans ++ ok")
print(str(round(np.mean(total_iteration),2))+" iterations")
print(str(kmeans.inertia_)+" inertia")


# data vizualisation of k-means ++ clustering results
fig = plt.pyplot.figure(num=None, figsize=(10, 10), dpi=200, facecolor='w', edgecolor='k')
ax = fig.add_subplot(111)

x =[]
y =[]

for i in range(0,len(X)):
    x.append(X[i][0])
    y.append(X[i][1])

color = kmeans.labels_
ax.scatter(x, y, s=50, c=color, marker='o',alpha=0.5)
centroids = kmeans.cluster_centers_
ax.scatter(centroids[:, 0], centroids[:, 1],marker='x', s=150, linewidths=3,color='k', zorder=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')


# populate clusters resulting from our kmeans++ algorithm using labels
clusters = [[] for _ in range(len(kmeans.cluster_centers_))]

for i in range(0,len(kmeans.labels_)):
    for label in range(0,len(kmeans.cluster_centers_)):
        if kmeans.labels_[i] == label:
            clusters[label].append(X[i])

print(str(len(clusters))+" clusters")
print("***")
for i in range(0,len(clusters)):
    print("cluster "+str(i)+" - "+str(len(clusters[i]))+" points centered on "+str(kmeans.cluster_centers_[i]))
    
    
    
# 2. DBVI Clustering
# Algorithm implementation following Davoud Moulavi - Pablo A. Jaskowiak† - Ricardo J. G. B. - Campello† Arthur Zimek† - Jörg Sander work.

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

DBVI = {}

# core distance definition
def core_dist(cluster,pos_clus):
    active_cluster = list(cluster)
    x = active_cluster.pop(pos_clus)
    knn_sum = 0
    for i in range(0,len(active_cluster)):
        knn_sum += (1/np.sqrt((x[0]-active_cluster[i][0])**2+(x[1]-active_cluster[i][1])**2))**d
    core_dist = (float(knn_sum)/float(len(active_cluster)))**(-1/d)
    return core_dist


# MR distance definition
def mr_distance(cluster,pos_clus_x,pos_clus_y):
    active_cluster = list(cluster)
    core_dist_x = core_dist(cluster,pos_clus_x)
    core_dist_y = core_dist(cluster,pos_clus_y)
    euc_dist_xy = np.sqrt((active_cluster[pos_clus_x][0]-active_cluster[pos_clus_y][0])**2+(active_cluster[pos_clus_x][1]-active_cluster[pos_clus_y][1])**2)
    return max(core_dist_x,core_dist_y,euc_dist_xy)

# MR Graphs definition using networkx
# MR graph is undirected, complete - nodes correpond to data points in cluster, weights to MR distance
# Note: we can verify that we get n(n-1)/2 arcs for each complete graph
def create_mr_graph(clusters):
    for k in range(0,len(clusters)):
        G = nx.Graph()
        for i in range(0,len(clusters[k])):
            G.add_node(i,pos=[clusters[k][i][0],clusters[k][i][1]])
            #pos=nx.get_node_attributes(G,'pos')

        for j in range(0,len(clusters[k])):
            for i in range(1+j,len(clusters[k])):
                G.add_edge(j, i, weight = mr_distance(clusters[k],j,i))
        DBVI["cluster%s"%k] = [G]

create_mr_graph(clusters)


# compute Density Sparsness of Cluster as the max weight of our MST
def DSC(DBVI):
    for k in range(0,len(DBVI)):
        # 1. minimum spanning tree
        min_tree = nx.minimum_spanning_tree(DBVI["cluster%s"%k][0])

        # 2. compute maximum weight of this MST
        max_weight = 0
        ind_max = -1
        for j in range(0,len(min_tree.edges())):
            if min_tree.edges(data=True)[j][2]['weight'] > max_weight:
                max_weight = min_tree.edges(data=True)[j][2]['weight']
                ind_max = j
        DSC = max_weight
        
        # 3. store results in existing dictionary
        DBVI["cluster%s"%k].append(min_tree)
        DBVI["cluster%s"%k].append(max_weight)

        
DSC(DBVI)


# compute Density Separation Between Pairs of Clusters (DSPC)

def DSPC(DBVI):
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
        
    # 3. compute core distance of each internal point of each cluster and append it to point lists
        for j in range(0,len(DBVI["cluster%s"%k][4])):
            DBVI["cluster%s"%k][4][j].append(core_dist(DBVI["cluster%s"%k][4],j))
            
    # 4. create variables in dictionary for smallest mr distances to other clusters
        smallest_mr_dist = []
        for l in range(0,len(DBVI)):
            smallest_mr_dist.append(0)
        DBVI["cluster%s"%k].append(smallest_mr_dist)
            
    # 5. compute density separation for each cluster - iterate on each point of each clusters 2 by 2
    for k1 in range(0,len(DBVI)):
        for k2 in range(1+k1,len(DVBI)):
            dspc = 1000000
            for p1 in range(0,len(DBVI["cluster%s"%k1][4])):
                for p2 in range(0,len(DBVI["cluster%s"%k2][4])):
                    euc_dist = np.sqrt((DBVI["cluster%s"%k1][4][p1][0]-DBVI["cluster%s"%k2][4][p2][0])**2+(DBVI["cluster%s"%k1][4][p1][1]-DBVI["cluster%s"%k2][4][p2][1])**2)
                    mr_dist = max(euc_dist,DBVI["cluster%s"%k1][4][p1][2],DBVI["cluster%s"%k2][4][p2][2])
                    if mr_dist < dspc:
                        dspc = mr_dist
            DBVI["cluster%s"%k1][5][k2] = dspc
            DBVI["cluster%s"%k2][5][k1] = dspc
                
DSPC(DBVI)


# compute Validity Index of each cluster
def VIC(DBVI):
    for k in range(0,len(DBVI)):
        min_dspc = min(i for i in DBVI["cluster%s"%k][5] if i > 0)
        dsc = DBVI["cluster%s"%k][2]
        V = (min_dspc-dsc)/(max(min_dspc,dsc))
        DBVI["cluster%s"%k].append(V)
        
        
VIC(DBVI)
